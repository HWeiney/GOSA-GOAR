import pytest
import torch

from swift.model.internvl_extensions.goar import GOSAOCRAnchorRefinement, find_goar_host, get_active_goar_adapter
from swift.template.templates.internvl import Internvl2Template


def _sample_data(target=True):
    return {
        'ocr': {
            'words': ['name', 'date'],
            'bboxes': [[0, 0, 500, 500], [500, 500, 1000, 1000]],
            'confidences': [0.95, 0.80],
        },
        'line_spans': [[2, 4], [5, 7]],
        'target': [0, 0, 500, 500] if target else [],
        'num_tiles': 1,
    }


def test_goar_roi_pool_respects_page_coordinates():
    # Token order for a 2x2 grid is top-left, top-right, bottom-left,
    # bottom-right. Each quadrant box should select exactly one token.
    visual = torch.arange(4, dtype=torch.float32).reshape(1, 4, 1)
    tile = torch.tensor([[0., 0., 1., 1.]])
    boxes = torch.tensor([[0., 0., 0.5, 0.5], [0.5, 0.5, 1., 1.]])
    pooled = GOSAOCRAnchorRefinement._roi_pool(visual, tile, boxes)
    torch.testing.assert_close(pooled[:, 0], torch.tensor([0., 3.]))


def test_goar_is_noop_at_initialization_and_has_finite_aux_loss():
    torch.manual_seed(7)
    module = GOSAOCRAnchorRefinement(hidden_size=16, bottleneck=8)
    embeds = torch.randn(1, 10, 16, requires_grad=True)
    labels = torch.full((1, 10), -100, dtype=torch.long)
    labels[:, 8:] = 1
    visual = torch.randn(1, 4, 16, requires_grad=True)
    tile = torch.tensor([[0., 0., 1., 1.]])
    visual_mask = torch.zeros(1, 10, dtype=torch.bool)
    visual_mask[:, :2] = True

    output, loss, boxes = module(embeds, [_sample_data()], visual, tile, labels, visual_mask)

    torch.testing.assert_close(output, embeds)
    assert loss is not None and torch.isfinite(loss)
    assert boxes[0] is not None and boxes[0].shape == (4, )
    assert torch.all((boxes[0] >= 0.) & (boxes[0] <= 1.))
    loss.backward()
    assert module.visual_box_head[-1].weight.grad is not None
    assert module.anchor_delta_head[-1].weight.grad is not None
    assert module.pointer_key.weight.grad is not None


def test_goar_inference_without_target_only_injects_prompt_features():
    torch.manual_seed(11)
    module = GOSAOCRAnchorRefinement(hidden_size=12, bottleneck=6)
    # Make the structural injection observable without training a full model.
    torch.nn.init.normal_(module.spatial_out[-1].weight, std=0.02)
    torch.nn.init.normal_(module.evidence_out.weight, std=0.02)
    embeds = torch.randn(1, 9, 12)
    visual = torch.randn(1, 4, 12)
    tile = torch.tensor([[0., 0., 1., 1.]])

    output, loss, boxes = module(embeds, [_sample_data(target=False)], visual, tile)

    assert loss is None
    assert boxes[0] is not None
    assert not torch.equal(output, embeds)
    # Non-OCR content before the final prompt token remains untouched.
    torch.testing.assert_close(output[:, 0], embeds[:, 0])


def test_goar_empty_ocr_uses_supervised_visual_fallback():
    torch.manual_seed(13)
    module = GOSAOCRAnchorRefinement(hidden_size=12, bottleneck=6)
    embeds = torch.randn(1, 9, 12, requires_grad=True)
    labels = torch.full((1, 9), -100, dtype=torch.long)
    labels[:, 7:] = 1
    visual = torch.randn(1, 4, 12, requires_grad=True)
    tile = torch.tensor([[0., 0., 1., 1.]])
    empty_data = {
        'ocr': {'words': [], 'bboxes': [], 'confidences': []},
        'line_spans': [],
        'target': [100, 100, 600, 600],
        'num_tiles': 1,
    }

    output, loss, boxes = module(embeds, [empty_data], visual, tile, labels)

    torch.testing.assert_close(output, embeds)
    assert loss is not None and torch.isfinite(loss)
    assert boxes[0] is not None and boxes[0].shape == (4, )
    loss.backward()
    assert module.visual_box_head[-1].weight.grad is not None
    assert module.anchor_delta_head[-1].weight.grad is None


def test_goar_nonempty_ocr_without_valid_spans_keeps_integrity_failure_signal():
    module = GOSAOCRAnchorRefinement(hidden_size=8, bottleneck=4)
    embeds = torch.randn(1, 9, 8)
    visual = torch.randn(1, 4, 8)
    tile = torch.tensor([[0., 0., 1., 1.]])
    invalid_data = _sample_data()
    invalid_data['line_spans'] = [[-1, -1], [-1, -1]]

    output, loss, boxes = module(embeds, [invalid_data], visual, tile)

    torch.testing.assert_close(output, embeds)
    assert loss is None
    assert boxes == [None]


def test_goar_same_checkpoint_selects_visual_ocr_or_fusion_box():
    torch.manual_seed(19)
    module = GOSAOCRAnchorRefinement(hidden_size=12, bottleneck=6)
    module.eval()
    with torch.no_grad():
        module.visual_box_head[-1].weight.zero_()
        module.visual_box_head[-1].bias.copy_(torch.tensor([-1.4, -1.0, -1.8, -1.6]))
        module.anchor_delta_head[-1].weight.zero_()
        module.anchor_delta_head[-1].bias.zero_()
        module.precision_head[-1].weight.zero_()
        module.precision_head[-1].bias.zero_()

    embeds = torch.randn(1, 9, 12)
    visual = torch.randn(1, 4, 12)
    tile = torch.tensor([[0., 0., 1., 1.]])
    selected = {}
    for mode in module.INFERENCE_BOX_MODES:
        module.inference_box_mode = mode
        _, loss, boxes = module(embeds, [_sample_data(target=False)], visual, tile)
        assert loss is None
        selected[mode] = boxes[0]

    assert not torch.allclose(selected['visual'], selected['ocr'])
    assert not torch.allclose(selected['fusion'], selected['visual'])
    assert not torch.allclose(selected['fusion'], selected['ocr'])


def test_goar_non_fusion_modes_are_rejected_during_training():
    module = GOSAOCRAnchorRefinement(hidden_size=8, bottleneck=4, inference_box_mode='visual')
    embeds = torch.randn(1, 9, 8)
    visual = torch.randn(1, 4, 8)
    tile = torch.tensor([[0., 0., 1., 1.]])

    with pytest.raises(ValueError, match='inference-only'):
        module(embeds, [_sample_data()], visual, tile)


def test_goar_without_roi_feature_is_invariant_to_local_token_permutation():
    torch.manual_seed(23)
    module = GOSAOCRAnchorRefinement(hidden_size=8, bottleneck=4, use_roi_feature=False)
    module.eval()
    embeds = torch.randn(1, 9, 8)
    visual = torch.randn(1, 4, 8)
    permuted = visual[:, torch.tensor([3, 2, 1, 0])]
    tile = torch.tensor([[0., 0., 1., 1.]])

    output_a, _, boxes_a = module(embeds, [_sample_data(target=False)], visual, tile)
    output_b, _, boxes_b = module(embeds, [_sample_data(target=False)], permuted, tile)

    torch.testing.assert_close(output_a, output_b)
    torch.testing.assert_close(boxes_a[0], boxes_b[0])


def test_goar_rejects_unknown_inference_box_mode():
    with pytest.raises(ValueError, match='GOAR_INFERENCE_BOX_MODE'):
        GOSAOCRAnchorRefinement(hidden_size=8, bottleneck=4, inference_box_mode='unknown')


def test_goar_fixed_mean_fusion_averages_visual_and_ocr_boxes():
    torch.manual_seed(29)
    module = GOSAOCRAnchorRefinement(hidden_size=12, bottleneck=6, source_fusion_mode='mean')
    module.eval()
    embeds = torch.randn(1, 9, 12)
    visual = torch.randn(1, 4, 12)
    tile = torch.tensor([[0., 0., 1., 1.]])
    selected = {}
    for mode in module.INFERENCE_BOX_MODES:
        module.inference_box_mode = mode
        _, _, boxes = module(embeds, [_sample_data(target=False)], visual, tile)
        selected[mode] = boxes[0]

    torch.testing.assert_close(selected['fusion'], (selected['visual'] + selected['ocr']) / 2.)


def test_goar_uncorrected_anchor_uses_pointer_weighted_raw_box():
    torch.manual_seed(31)
    module = GOSAOCRAnchorRefinement(hidden_size=12, bottleneck=6, refine_ocr_anchor=False)
    module.eval()
    module.inference_box_mode = 'ocr'
    with torch.no_grad():
        module.pointer_key.weight.zero_()
        module.pointer_query.weight.zero_()
        module.anchor_delta_head[-1].bias.fill_(1.)
    embeds = torch.randn(1, 9, 12)
    visual = torch.randn(1, 4, 12)
    tile = torch.tensor([[0., 0., 1., 1.]])

    _, _, boxes = module(embeds, [_sample_data(target=False)], visual, tile)

    torch.testing.assert_close(boxes[0], torch.tensor([.25, .25, .75, .75]))


def test_goar_rejects_unknown_source_fusion_mode():
    with pytest.raises(ValueError, match='GOAR_SOURCE_FUSION_MODE'):
        GOSAOCRAnchorRefinement(hidden_size=8, bottleneck=4, source_fusion_mode='unknown')


class _DelegatingWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def test_find_goar_host_ignores_delegated_wrapper_attributes():
    inner = torch.nn.Module()
    inner.goar_adapter = GOSAOCRAnchorRefinement(hidden_size=8, bottleneck=4)
    outer = _DelegatingWrapper(inner)

    assert hasattr(outer, 'goar_adapter')
    assert find_goar_host(outer) is inner


class _ModulesToSaveLike(torch.nn.Module):

    def __init__(self, original, trainable):
        super().__init__()
        self.original_module = original
        self.modules_to_save = torch.nn.ModuleDict({'default': trainable})
        self.active_adapters = ['default']


def test_get_active_goar_adapter_selects_trainable_peft_copy():
    host = torch.nn.Module()
    original = GOSAOCRAnchorRefinement(hidden_size=8, bottleneck=4)
    trainable = GOSAOCRAnchorRefinement(hidden_size=8, bottleneck=4)
    host.goar_adapter = _ModulesToSaveLike(original, trainable)

    assert get_active_goar_adapter(host) is trainable


class _CharProcessor:

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(char) for char in text]


class _MergedLineEndProcessor:

    @staticmethod
    def encode(text, add_special_tokens=False):
        tokens, index = [], 0
        while index < len(text):
            if text.startswith(')\n', index):
                tokens.append(0x110000)
                index += 2
            else:
                tokens.append(ord(text[index]))
                index += 1
        return tokens


def test_goar_ocr_line_spans_preserve_row_alignment():
    template = Internvl2Template.__new__(Internvl2Template)
    template.processor = _CharProcessor()
    record = {
        'words': ['first', 'second'],
        'bboxes': [[1, 2, 3, 4], [5, 6, 7, 8]],
        'confidences': [0.9, 0.8],
    }
    prompt = Internvl2Template._ocr_prompt(record)
    input_ids = _CharProcessor.encode('prefix' + prompt + 'suffix')

    spans = template._find_ocr_line_spans(input_ids, record)

    assert len(spans) == 2
    decoded = [''.join(chr(value) for value in input_ids[start:end]) for start, end in spans]
    assert decoded[0].startswith('first <box>[[1,2,3,4]]')
    assert decoded[1].startswith('second <box>[[5,6,7,8]]')


def test_goar_ocr_line_spans_include_tokenizer_merged_line_end():
    template = Internvl2Template.__new__(Internvl2Template)
    template.processor = _MergedLineEndProcessor()
    record = {
        'words': ['first', 'second'],
        'bboxes': [[1, 2, 3, 4], [5, 6, 7, 8]],
        'confidences': [0.9, 0.8],
    }
    input_ids = template.processor.encode(template._ocr_prompt(record), add_special_tokens=False)

    spans = template._find_ocr_line_spans(input_ids, record)

    assert all(span != [-1, -1] for span in spans)
    assert all(input_ids[end - 1] == 0x110000 for _, end in spans)
