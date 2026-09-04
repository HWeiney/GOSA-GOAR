from tools.ocr_evidence.convert_sroie import (
    align_entity_bbox, align_total, build_field_row, freeze_split, normalize_bbox,
)


def test_normalize_sroie_bbox():
    assert normalize_bbox([10, 20, 30, 40], 100, 200) == [100, 100, 300, 200]


def test_multiline_address_alignment_uses_contiguous_regions():
    words = ['SHOP SDN BHD', 'NO 2 & 4', 'JALAN BAYU 4', 'BANDAR SERI ALAM',
             'TEL 123', 'TOTAL']
    boxes = [[0, i * 10, 100, (i + 1) * 10] for i in range(len(words))]
    bbox, audit = align_entity_bbox(
        'address', 'NO 2 & 4, JALAN BAYU 4, BANDAR SERI ALAM', words, boxes)
    assert audit['ocr_indices'] == [1, 2, 3]
    assert bbox == [0, 10, 100, 40]
    assert audit['token_recall'] == 1.0


def test_total_alignment_uses_standalone_total_anchor():
    words = ['ITEM', '193.00', 'TOTAL EXCLUDE GST:', '193.00', 'TOTAL:', '193.00',
             'VISA CARD', '193.00']
    indices, audit = align_total(words, '193.00')
    assert indices == [5]
    assert audit['total_anchor_index'] == 4


def test_total_alignment_does_not_treat_punctuation_as_amount():
    words = ['TOTAL:', ':', '%', '2', '$278.80']
    indices, _ = align_total(words, '278.80')
    assert indices == [4]


def test_numeric_date_alignment_accepts_annotation_order():
    words = ['SHOP', '04/03/2018 15:41:52', 'TOTAL']
    boxes = [[0, i * 10, 100, (i + 1) * 10] for i in range(len(words))]
    _, audit = align_entity_bbox('date', '20180304', words, boxes)
    assert audit['ocr_indices'] == [1]
    assert audit['method'] == 'date-equivalent-digit-order'


def test_frozen_sroie_split_is_deterministic_and_disjoint():
    train_keys = [f'train-{index}' for index in range(20)]
    test_keys = [f'test-{index}' for index in range(5)]
    first = freeze_split(train_keys, test_keys, seed=42, train_size=8, valid_size=4)
    second = freeze_split(train_keys, test_keys, seed=42, train_size=8, valid_size=4)
    assert first == second
    assert len(first['train']) == 8
    assert len(first['valid']) == 4
    assert set(first['train']).isdisjoint(first['valid'])
    assert set(first['train']).isdisjoint(first['test'])


def test_missing_kie_annotation_is_skipped(tmp_path):
    source = {'key': 'missing', 'entities': {'total': ''}}
    assert build_field_row(
        source, split='train', field_name='total', image_path=tmp_path / 'x.jpg',
        cache_path=tmp_path / 'x.json', width=100, height=100,
        words=['TOTAL'], boxes=[[0, 0, 1000, 1000]]) is None
