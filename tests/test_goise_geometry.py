import pathlib
import warnings

import pytest
import torch
from types import SimpleNamespace
from PIL import Image

from swift.template.templates.internvl import Internvl2Template
from swift.template.vision_utils import _dynamic_preprocess
from swift.model.internvl_extensions.gosa import GlobalTilePositionalEncoding, attach_gosa


def test_dynamic_preprocess_returns_normalized_page_geometry():
    image = Image.new('RGB', (400, 100))
    tiles, tile_coords, tile_types = _dynamic_preprocess(
        image, max_num=4, image_size=32, use_thumbnail=True, return_tile_coords=True)

    assert len(tiles) == 5
    torch.testing.assert_close(tile_coords[0], torch.tensor([0., 0., 0.25, 1.]))
    torch.testing.assert_close(tile_coords[3], torch.tensor([0.75, 0., 1., 1.]))
    torch.testing.assert_close(tile_coords[4], torch.tensor([0., 0., 1., 1.]))
    torch.testing.assert_close(tile_types, torch.tensor([0, 0, 0, 0, 1]))


def test_single_full_page_tile_is_not_marked_as_thumbnail():
    image = Image.new('RGB', (100, 100))
    tiles, tile_coords, tile_types = _dynamic_preprocess(
        image, max_num=1, image_size=32, use_thumbnail=True, return_tile_coords=True)

    assert len(tiles) == 1
    torch.testing.assert_close(tile_coords, torch.tensor([[0., 0., 1., 1.]]))
    torch.testing.assert_close(tile_types, torch.tensor([0]))


def _make_encoder(fusion='add', thumbnail='full', gate_init=0.1, coord_mode='real'):
    encoder = GlobalTilePositionalEncoding(
        dim=4, num_bands=1, fusion=fusion, gate_init=gate_init, thumbnail_mode=thumbnail,
        coord_mode=coord_mode)
    with torch.no_grad():
        encoder.mlp[-1].weight.fill_(0.25)
        encoder.mlp[-1].bias.fill_(0.1)
    return encoder


def test_pretrained_model_source_is_a_clean_baseline():
    model_source = (pathlib.Path(__file__).parents[1] / 'pretrained' / 'InternVL3_5-2B'
                    / 'modeling_internvl_chat.py')
    if not model_source.is_file():
        pytest.skip('The optional pretrained checkpoint is not present.')
    source = model_source.read_text(encoding='utf-8')
    assert 'GlobalTilePositionalEncoding' not in source
    assert 'Learnable2DPositionalEncoding' not in source
    assert 'ENABLE_GLOBAL_PE' not in source
    assert 'ENABLE_LOCAL_PE' not in source


def test_gosa_is_attached_only_when_enabled(monkeypatch):
    class TinyInternVL(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(vision_config=SimpleNamespace(hidden_size=1))
            self.downsample_ratio = .5
            self.mlp1 = torch.nn.Sequential(torch.nn.Linear(4, 4))

        def extract_feature(self, pixel_values):
            return pixel_values

    monkeypatch.delenv('ENABLE_LOCAL_PE', raising=False)
    monkeypatch.delenv('ENABLE_GLOBAL_PE', raising=False)
    baseline = TinyInternVL()
    assert attach_gosa(baseline) is False
    assert not hasattr(baseline, 'pos_embed_2d')
    assert not hasattr(baseline, 'global_spatial_encoder')

    monkeypatch.setenv('ENABLE_LOCAL_PE', 'True')
    monkeypatch.setenv('ENABLE_GLOBAL_PE', 'True')
    enhanced = TinyInternVL()
    assert attach_gosa(enhanced) is True
    assert enhanced.pos_embed_2d.row_embed.shape == (64, 2)
    assert enhanced.global_spatial_encoder.mlp[-1].out_features == 4


def test_add_fusion_matches_dense_global_encoding():
    encoder = _make_encoder(fusion='add')
    boxes = torch.tensor([[0., 0., 0.5, 1.]])
    tile_types = torch.tensor([0])

    dense = encoder._encode_positions(boxes, 2, 2, torch.float32, boxes.device)
    fused = encoder(boxes, tile_types, 2, 2, torch.float32, boxes.device)

    torch.testing.assert_close(fused, dense)


def test_full_page_coord_mode_removes_actual_crop_geometry_without_changing_parameters():
    encoder = _make_encoder(fusion='add', coord_mode='full_page')
    crop_box = torch.tensor([[0., 0., 0.5, 1.]])
    full_page_box = torch.tensor([[0., 0., 1., 1.]])

    ablated = encoder(crop_box, torch.tensor([0]), 2, 2, torch.float32, crop_box.device)
    expected = encoder._encode_positions(full_page_box, 2, 2, torch.float32, crop_box.device)
    real_geometry = encoder._encode_positions(crop_box, 2, 2, torch.float32, crop_box.device)

    torch.testing.assert_close(ablated, expected)
    assert not torch.allclose(ablated, real_geometry)


def test_uniform_coord_mode_uses_page_independent_near_square_grid():
    encoder = _make_encoder(fusion='add', coord_mode='uniform')
    boxes = torch.tensor([
        [0., 0., 0.25, 1.], [0.25, 0., 0.5, 1.],
        [0.5, 0., 0.75, 1.], [0.75, 0., 1., 1.],
        [0., 0., 1., 1.],
    ])
    tile_types = torch.tensor([0, 0, 0, 0, 1])

    transformed = encoder._coordinate_ablation(boxes, tile_types)

    expected = torch.tensor([
        [0., 0., 0.5, 0.5], [0.5, 0., 1., 0.5],
        [0., 0.5, 0.5, 1.], [0.5, 0.5, 1., 1.],
        [0., 0., 1., 1.],
    ])
    torch.testing.assert_close(transformed, expected)


def test_shuffled_coord_mode_is_seeded_and_keeps_thumbnail_fixed():
    boxes = torch.tensor([
        [0., 0., 0.25, 1.], [0.25, 0., 0.5, 1.],
        [0.5, 0., 0.75, 1.], [0.75, 0., 1., 1.],
        [0., 0., 1., 1.],
    ])
    tile_types = torch.tensor([0, 0, 0, 0, 1])
    first = GlobalTilePositionalEncoding(dim=4, num_bands=1, coord_mode='shuffled', coord_seed=17)
    second = GlobalTilePositionalEncoding(dim=4, num_bands=1, coord_mode='shuffled', coord_seed=17)

    transformed = first._coordinate_ablation(boxes, tile_types)

    torch.testing.assert_close(transformed, second._coordinate_ablation(boxes, tile_types))
    torch.testing.assert_close(transformed[-1], boxes[-1])
    assert torch.all(torch.any(transformed[:-1] != boxes[:-1], dim=1))
    assert {tuple(row.tolist()) for row in transformed[:-1]} == {tuple(row.tolist()) for row in boxes[:-1]}


def test_shuffled_coord_mode_deranges_two_tile_pages():
    boxes = torch.tensor([[0., 0., 1., 0.5], [0., 0.5, 1., 1.]])
    encoder = GlobalTilePositionalEncoding(dim=4, num_bands=1, coord_mode='shuffled', coord_seed=0)

    transformed = encoder._coordinate_ablation(boxes, torch.tensor([0, 0]))

    torch.testing.assert_close(transformed, boxes.flip(0))


def test_disabled_coord_mode_is_zero_but_keeps_parameters_in_graph():
    encoder = _make_encoder(fusion='add', coord_mode='disabled')
    boxes = torch.tensor([[0., 0., 0.5, 1.], [0., 0., 1., 1.]])
    output = encoder(boxes, torch.tensor([0, 1]), 2, 2, torch.float32, boxes.device)

    torch.testing.assert_close(output, torch.zeros_like(output))
    output.sum().backward()
    assert encoder.mlp[-1].weight.grad is not None


def test_scalar_and_channel_gates_start_at_requested_scale():
    boxes = torch.tensor([[0., 0., 0.5, 1.]])
    tile_types = torch.tensor([0])
    for fusion in ('scalar_gate', 'channel_gate'):
        encoder = _make_encoder(fusion=fusion, gate_init=0.1)
        dense = encoder._encode_positions(boxes, 2, 2, torch.float32, boxes.device)
        fused = encoder(boxes, tile_types, 2, 2, torch.float32, boxes.device)
        torch.testing.assert_close(fused, dense * 0.1, rtol=1e-5, atol=1e-6)


def test_thumbnail_type_only_omits_dense_global_encoding():
    encoder = _make_encoder(fusion='add', thumbnail='type_only')
    with torch.no_grad():
        encoder.tile_type_embedding.weight[1].fill_(2.)
    boxes = torch.tensor([[0., 0., 1., 1.]])

    fused = encoder(boxes, torch.tensor([1]), 2, 2, torch.float32, boxes.device)

    torch.testing.assert_close(fused, torch.full_like(fused, 2.))


def test_thumbnail_off_has_no_added_position_features():
    encoder = _make_encoder(fusion='channel_gate', thumbnail='off')
    with torch.no_grad():
        encoder.tile_type_embedding.weight[1].fill_(2.)
    boxes = torch.tensor([[0., 0., 1., 1.]])

    fused = encoder(boxes, torch.tensor([1]), 2, 2, torch.float32, boxes.device)

    torch.testing.assert_close(fused, torch.zeros_like(fused))


def test_missing_tile_types_falls_back_to_regular_tiles_with_warning():
    encoder = _make_encoder(fusion='add')
    boxes = torch.tensor([[0., 0., 1., 1.]])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fallback = encoder(boxes, None, 2, 2, torch.float32, boxes.device)
        encoder(boxes, None, 2, 2, torch.float32, boxes.device)
    expected = encoder(boxes, torch.tensor([0]), 2, 2, torch.float32, boxes.device)

    torch.testing.assert_close(fallback, expected)
    assert sum('tile_types' in str(item.message) for item in caught) == 1


def test_global_encoder_and_gate_receive_gradients():
    encoder = _make_encoder(fusion='channel_gate')
    boxes = torch.tensor([[0., 0., 0.5, 1.]])
    output = encoder(boxes, torch.tensor([0]), 2, 2, torch.float32, boxes.device)

    output.sum().backward()

    assert encoder.mlp[-1].weight.grad is not None
    assert encoder.gate_logit.grad is not None


def test_channel_gate_sanitizes_overflowing_broadcast_gradient():
    encoder = _make_encoder(fusion='channel_gate')
    boxes = torch.tensor([[0., 0., 0.5, 1.]])
    output = encoder(boxes, torch.tensor([0]), 2, 2, torch.float32, boxes.device)

    output.backward(torch.full_like(output, float('inf')))

    assert torch.isfinite(encoder.gate_logit.grad).all()


class _CharProcessor:

    @staticmethod
    def encode(text, add_special_tokens=False):
        return [ord(ch) for ch in text]


def test_grounding_token_weight_scales_only_box_coordinate_tokens():
    template = Internvl2Template.__new__(Internvl2Template)
    template.processor = _CharProcessor()
    template.grounding_box_loss_weight = 1.5
    template.grounding_ref_loss_weight = 1.0
    answer = 'prefix <ref>ABC</ref><box>[[1,2,3,4]]</box> suffix'
    encoded = {
        'input_ids': _CharProcessor.encode(answer),
        'labels': _CharProcessor.encode(answer),
        'loss_scale': [1.0] * len(answer),
    }

    changed = template._apply_grounding_token_weights(encoded, 'ABC', '1,2,3,4')

    assert changed is True
    box_start = answer.index('1,2,3,4')
    ref_start = answer.index('ABC')
    assert encoded['loss_scale'][box_start:box_start + len('1,2,3,4')] == [1.5] * len('1,2,3,4')
    assert encoded['loss_scale'][ref_start:ref_start + len('ABC')] == [1.0] * len('ABC')
    assert encoded['loss_scale'][box_start - 1] == 1.0
    assert encoded['loss_scale'][box_start + len('1,2,3,4')] == 1.0


def test_grounding_token_weight_creates_loss_scale_when_missing():
    template = Internvl2Template.__new__(Internvl2Template)
    template.processor = _CharProcessor()
    template.grounding_box_loss_weight = 1.5
    template.grounding_ref_loss_weight = 1.0
    answer = '<ref>ABC</ref><box>[[1,2,3,4]]</box>'
    input_ids = _CharProcessor.encode(answer)
    labels = [-100] * 5 + input_ids[5:]
    encoded = {'input_ids': input_ids, 'labels': labels}

    changed = template._apply_grounding_token_weights(encoded, 'ABC', '1,2,3,4')

    assert changed is True
    box_start = answer.index('1,2,3,4')
    assert encoded['loss_scale'][:5] == [0.0] * 5
    assert encoded['loss_scale'][box_start:box_start + len('1,2,3,4')] == [1.5] * len('1,2,3,4')


def test_grounding_token_weight_multiplies_existing_loss_scale():
    template = Internvl2Template.__new__(Internvl2Template)
    template.processor = _CharProcessor()
    template.grounding_box_loss_weight = 1.5
    template.grounding_ref_loss_weight = 1.0
    answer = '<ref>ABC</ref><box>[[1,2,3,4]]</box>'
    encoded = {
        'input_ids': _CharProcessor.encode(answer),
        'labels': _CharProcessor.encode(answer),
        'loss_scale': [2.0] * len(answer),
    }

    changed = template._apply_grounding_token_weights(encoded, 'ABC', '1,2,3,4')

    assert changed is True
    box_start = answer.index('1,2,3,4')
    assert encoded['loss_scale'][box_start:box_start + len('1,2,3,4')] == [3.0] * len('1,2,3,4')


def test_grounding_token_weight_returns_false_when_box_span_missing():
    template = Internvl2Template.__new__(Internvl2Template)
    template.processor = _CharProcessor()
    template.grounding_box_loss_weight = 1.5
    template.grounding_ref_loss_weight = 1.0
    encoded = {
        'input_ids': _CharProcessor.encode('<ref>ABC</ref>'),
        'labels': _CharProcessor.encode('<ref>ABC</ref>'),
        'loss_scale': [1.0] * len('<ref>ABC</ref>'),
    }
    original_loss_scale = list(encoded['loss_scale'])

    changed = template._apply_grounding_token_weights(encoded, 'ABC', '1,2,3,4')

    assert changed is False
    assert encoded['loss_scale'] == original_loss_scale


def test_add_grounding_target_ignores_malformed_box_text():
    template = Internvl2Template.__new__(Internvl2Template)
    template.processor = _CharProcessor()
    template.enable_grounding_token_weight = True
    template.grounding_box_loss_weight = 1.5
    template.grounding_ref_loss_weight = 1.0
    answer = '<ref>ABC</ref><box>[[bad,2,3,4]]</box>'
    encoded = {
        'input_ids': _CharProcessor.encode(answer),
        'labels': _CharProcessor.encode(answer),
        'loss_scale': [1.0] * len(answer),
    }
    inputs = SimpleNamespace(
        images=[],
        messages=[{
            'role': 'assistant',
            'content': answer,
        }])

    template._add_grounding_target(encoded, inputs)

    assert encoded['loss_scale'] == [1.0] * len(answer)


def test_add_grounding_target_creates_default_loss_scale_without_box_when_enabled():
    template = Internvl2Template.__new__(Internvl2Template)
    template.processor = _CharProcessor()
    template.enable_grounding_token_weight = True
    template.grounding_box_loss_weight = 1.5
    template.grounding_ref_loss_weight = 1.0
    answer = 'plain answer without grounding markup'
    input_ids = _CharProcessor.encode(answer)
    labels = [-100] * 6 + input_ids[6:]
    encoded = {'input_ids': input_ids, 'labels': labels}
    inputs = SimpleNamespace(
        images=[],
        messages=[{
            'role': 'assistant',
            'content': answer,
        }])

    template._add_grounding_target(encoded, inputs)

    assert encoded['loss_scale'] == [0.0] * 6 + [1.0] * (len(answer) - 6)
