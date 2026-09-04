# Dataset preparation

The repository contains derived JSONL annotations for public datasets, but not
the source images. Always review and follow the original dataset license and
terms before downloading or redistributing data.

## SRFUND

Source: [SRFUND project page](https://sprateam-ustc.github.io/SRFUND/). The
project reports 1,592 fully annotated forms and provides the official access
point for the dataset.

Place the downloaded images in this layout:

```text
data/SRFUND/images/
├── en/
└── zh/
```

The frozen derived annotations are:

```text
annotations/SRFUND/srfund_en_train.jsonl
annotations/SRFUND/srfund_en_val.jsonl
annotations/SRFUND/srfund_zh_train.jsonl
annotations/SRFUND/srfund_zh_val.jsonl
```

They are produced from the official entity relations with
`tools/ocr_evidence/convert_srfund.py`. The conversion rules are:

- use one `Question` entity linked to one unambiguous `Answer` entity;
- merge multiple lines inside the same answer entity and use their union box;
- keep a uniquely checked option, but reject multiple checked options;
- reject table headers linked to multiple distinct rows;
- reject duplicate field text within one image because the prompt has no
  question-region identifier;
- remove decorative checkbox/field glyphs while preserving semantic
  punctuation;
- normalize `xyxy` boxes to integer page coordinates in `[0, 1000]`;
- keep the official English ordering (149 train documents, then 50 validation
  documents) and the official Chinese train/validation naming.

To regenerate the annotations from an extracted official copy:

```bash
python tools/ocr_evidence/convert_srfund.py \
  --source /path/to/SRFUND \
  --image-root data/SRFUND/images \
  --output-dir annotations/SRFUND
```

## SROIE

The reproducible source used here is the
[ICDAR2019-SROIE Hugging Face release](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE),
which documents a CC-BY-4.0 license and provides 626 training receipts plus an
extended test split. It can be downloaded with:

```bash
python - <<'PY'
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id='jsdnrs/ICDAR2019-SROIE',
    repo_type='dataset',
    local_dir='data/SROIE/source',
)
PY
```

Create the frozen 200/100/all split and proxy boxes:

```bash
python tools/ocr_evidence/convert_sroie.py \
  --source data/SROIE/source \
  --output-dir data/SROIE \
  --seed 42 --train-size 200 --valid-size 100
```

The checked-in JSONL files under `annotations/SROIE/` use the same frozen
sample keys and relative runtime paths. Conversion rules are:

- sample at receipt-image level so fields from one image never cross splits;
- emit independent rows for `company`, `date`, `address`, and `total`;
- skip empty labels instead of fabricating targets;
- align each KIE value to contiguous dataset-provided OCR regions;
- use the amount nearest the last standalone `TOTAL` anchor for ambiguous total
  values;
- normalize proxy boxes to integer `[0, 1000]` coordinates;
- record OCR indices, overlap, sequence similarity, and alignment score for
  every proxy box;
- impute OCR confidence as `1.0`, because the official OCR regions do not
  provide confidence values.

SROIE text metrics are ordinary KIE results. Mean IoU and grounded accuracy are
proxy-grounding metrics because SROIE does not directly link KIE values to OCR
boxes.

## Private dataset interface

No private image, file name, field value, source path, or annotation is
included. A compatible dataset only needs one JSON object per line with:

- `id`: an opaque sample identifier;
- `images`: one repository-relative image path;
- `messages`: one user prompt and one assistant target;
- `question`, `answer`, and `answer_bbox`;
- `ocr_cache` when OCR prompt or GOAR is enabled.

See `annotations/private_dataset/schema.json` and
`annotations/private_dataset/example.synthetic.jsonl`. The example is wholly
synthetic and is not derived from the private dataset.
