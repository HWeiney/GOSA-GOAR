#!/usr/bin/env python3
"""Run a pinned PP-OCR version once and cache normalized predictions as JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List

from PIL import Image


def polygon_to_xyxy(polygon) -> List[float]:
    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    return [min(xs), min(ys), max(xs), max(ys)]


def normalize_bbox(box, width: int, height: int) -> List[int]:
    result = [round(1000 * box[0] / width), round(1000 * box[1] / height),
              round(1000 * box[2] / width), round(1000 * box[3] / height)]
    return [max(0, min(1000, int(value))) for value in result]


def parse_paddle_result(result) -> Iterable[Dict]:
    """Accept both PaddleOCR 3.x ``predict`` and legacy ``ocr`` outputs."""
    if hasattr(result, 'json'):
        result = result.json
    if isinstance(result, dict):
        data = result.get('res', result)
        texts = data.get('rec_texts', [])
        scores = data.get('rec_scores', [])
        polygons = data.get('rec_polys', data.get('dt_polys', []))
        for text, score, polygon in zip(texts, scores, polygons):
            yield {'text': text, 'confidence': float(score), 'polygon': polygon}
        return
    lines = result
    if isinstance(lines, list) and len(lines) == 1 and isinstance(lines[0], list):
        lines = lines[0]
    for line in lines or []:
        if not isinstance(line, (list, tuple)) or len(line) < 2:
            continue
        polygon, recognition = line[0], line[1]
        if isinstance(recognition, (list, tuple)) and len(recognition) >= 2:
            yield {'text': recognition[0], 'confidence': float(recognition[1]), 'polygon': polygon}


def build_engine(language: str, args):
    try:
        from paddleocr import PaddleOCR
    except ImportError as exc:
        raise SystemExit('PaddleOCR is not installed. Install paddlepaddle and paddleocr in the OCR environment.') from exc
    lang = 'ch' if language == 'zh' else 'en'
    if args.ocr_version == 'PP-OCRv6':
        detection_model = 'PP-OCRv6_medium_det'
        recognition_model = 'PP-OCRv6_medium_rec'
    else:
        detection_model = 'PP-OCRv5_server_det'
        recognition_model = 'PP-OCRv5_server_rec' if language == 'zh' else 'en_PP-OCRv5_mobile_rec'
    new_kwargs = {}
    if args.text_detection_model_dir:
        new_kwargs['text_detection_model_dir'] = str(args.text_detection_model_dir)
    else:
        new_kwargs['text_detection_model_name'] = detection_model
    recognition_dir = args.zh_recognition_model_dir if language == 'zh' else args.en_recognition_model_dir
    if recognition_dir:
        new_kwargs['text_recognition_model_dir'] = str(recognition_dir)
    else:
        new_kwargs['text_recognition_model_name'] = recognition_model
    if args.device:
        new_kwargs['device'] = args.device
    try:
        engine = PaddleOCR(
            lang=lang,
            ocr_version=args.ocr_version,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            **new_kwargs,
        )
    except TypeError:  # legacy PaddleOCR constructor
        legacy_kwargs = {}
        if args.text_detection_model_dir:
            legacy_kwargs['det_model_dir'] = str(args.text_detection_model_dir)
        if recognition_dir:
            legacy_kwargs['rec_model_dir'] = str(recognition_dir)
        engine = PaddleOCR(lang=lang, use_angle_cls=False, show_log=False, **legacy_kwargs)
    return engine, detection_model, recognition_model


def predict(engine, image_path: str):
    if hasattr(engine, 'predict'):
        return list(engine.predict(input=image_path))
    return [engine.ocr(image_path, cls=False)]


def cache_image(engine, image_path: Path, cache_path: Path, language: str,
                ocr_version: str, detection_model: str, recognition_model: str) -> None:
    with Image.open(image_path) as image:
        width, height = image.size
    words, boxes, boxes_px, confidences = [], [], [], []
    for batch_result in predict(engine, str(image_path)):
        for item in parse_paddle_result(batch_result):
            text = str(item['text']).strip()
            if not text:
                continue
            box_px = polygon_to_xyxy(item['polygon'])
            words.append(text)
            boxes_px.append([round(value, 2) for value in box_px])
            boxes.append(normalize_bbox(box_px, width, height))
            confidences.append(max(0., min(1., float(item['confidence']))))
    payload = {
        'schema_version': 1,
        'engine': ocr_version,
        'detection_model': detection_model,
        'recognition_model': recognition_model,
        'source': str(image_path.resolve()),
        'image_size': [width, height],
        'bbox_space': '[0,1000]',
        'language': language,
        'words': words,
        'bboxes': boxes,
        'bboxes_px': boxes_px,
        'confidences': confidences,
    }
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')


def refresh_cache_source(cache_path: Path, image_path: Path) -> bool:
    """Keep cache provenance current without running OCR again."""
    payload = json.loads(cache_path.read_text(encoding='utf-8'))
    source = str(image_path.resolve())
    if payload.get('source') == source:
        return False
    payload['source'] = source
    cache_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-jsonl', type=Path, nargs='+', required=True)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--ocr-version', choices=['PP-OCRv5', 'PP-OCRv6'], default='PP-OCRv6')
    parser.add_argument('--text-detection-model-dir', type=Path,
                        help='Local PP-OCR detection model directory (recommended for offline execution).')
    parser.add_argument('--zh-recognition-model-dir', type=Path,
                        help='Local PP-OCR Chinese recognition model directory.')
    parser.add_argument('--en-recognition-model-dir', type=Path,
                        help='Local PP-OCR English recognition model directory.')
    parser.add_argument('--device',
                        help='PaddleOCR device, for example gpu:0 or cpu. Uses PaddleOCR default when omitted.')
    args = parser.parse_args()
    engines = {}
    seen = set()
    for dataset_path in args.dataset_jsonl:
        with dataset_path.open(encoding='utf-8') as f:
            for line in f:
                row = json.loads(line)
                image_path = Path(row['images'][0])
                cache_path = Path(row['ocr_cache'])
                key = str(cache_path.resolve())
                if key in seen:
                    seen.add(key)
                    continue
                if cache_path.exists() and not args.overwrite:
                    if refresh_cache_source(cache_path, image_path):
                        print(f'refreshed source: {cache_path}')
                    seen.add(key)
                    continue
                language = row.get('language', 'en')
                if language not in engines:
                    engines[language] = build_engine(language, args)
                engine, detection_model, recognition_model = engines[language]
                # Only the image path is passed to OCR. answer/answer_bbox and
                # all SRFUND entity annotations are never visible to the engine.
                cache_image(engine, image_path, cache_path, language, args.ocr_version,
                            detection_model, recognition_model)
                seen.add(key)
                print(cache_path)


if __name__ == '__main__':
    main()
