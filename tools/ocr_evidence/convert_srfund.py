#!/usr/bin/env python3
"""Convert SRFUND zh/en entity relations to grounded field instructions."""

from __future__ import annotations

import argparse
import io
import json
import re
import zipfile
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from PIL import Image


def union_entity_box(entity: Dict) -> List[float]:
    boxes = []
    for line in entity.get('lines') or []:
        words = [word.get('box') for word in line.get('words') or [] if word.get('box')]
        boxes.extend(words or ([line['box']] if line.get('box') else []))
    if not boxes and entity.get('box'):
        boxes = [entity['box']]
    if not boxes:
        raise ValueError(f'Entity {entity.get("id")} has no usable bbox.')
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def normalize_bbox(box: List[float], width: int, height: int) -> List[int]:
    values = [round(1000 * box[0] / width), round(1000 * box[1] / height),
              round(1000 * box[2] / width), round(1000 * box[3] / height)]
    values = [max(0, min(1000, int(value))) for value in values]
    values[0], values[2] = sorted((values[0], values[2]))
    values[1], values[3] = sorted((values[1], values[3]))
    return values


def linked_answers(question: Dict, entities_by_id: Dict[int, Dict]) -> Iterable[Dict]:
    seen = set()
    for pair in question.get('linking') or []:
        if len(pair) != 2 or question['id'] not in pair:
            continue
        other_id = pair[1] if pair[0] == question['id'] else pair[0]
        answer = entities_by_id.get(other_id)
        if answer and answer.get('label', '').lower() == 'answer' and other_id not in seen:
            seen.add(other_id)
            yield answer


SELECTION_MARKERS = '□☐☑☒✓✔√■●○◉'
SELECTED_MARKERS = '☑☒✓✔■●◉'


def clean_entity_text(text: str, *, question: bool = False) -> str:
    """Remove annotation/layout glyphs while preserving semantic punctuation."""
    text = re.sub(r'\s+', ' ', str(text)).strip()
    text = re.sub(f'[{re.escape(SELECTION_MARKERS)}]', '', text).strip()
    text = re.sub(r'^[*★☆▽▼△▲◆◇▪•_]+\s*', '', text).strip()
    if question:
        text = re.sub(r'\s*[:：;；,，。.!！?？*_]+\s*$', '', text).strip()
    return text


def is_selected_answer(answer: Dict) -> bool:
    text = str(answer.get('text', '')).strip()
    if any(marker in text for marker in SELECTED_MARKERS):
        return True
    # A standalone or short check mark is a selection, while instructional
    # prose that merely mentions a check mark is not.
    return '√' in text and len(text) <= 30 and not re.search(r'[请打划选]', text)


def select_and_order_answers(answers: Iterable[Dict]) -> Tuple[List[Dict], str]:
    """Resolve checkbox choices and reject unrelated one-to-many entities."""
    answers = list(answers)
    selected = [answer for answer in answers if is_selected_answer(answer)]
    if selected:
        if len(selected) > 1:
            return [], 'ambiguous_multiple_selected_answers'
        answers = selected
        policy = 'selected_options'
    elif len(answers) > 1:
        # A field header linked to several table rows is not a multi-line
        # answer. With image + field text alone, no row can be selected
        # unambiguously, so using a fabricated concatenation is harmful.
        return [], 'ambiguous_multiple_answers'
    else:
        policy = 'single'
    answers.sort(key=lambda answer: (
        union_entity_box(answer)[1], union_entity_box(answer)[0], int(answer['id'])))
    return answers, policy


def union_answer_boxes(answers: Iterable[Dict]) -> List[float]:
    boxes = [union_entity_box(answer) for answer in answers]
    return [min(box[0] for box in boxes), min(box[1] for box in boxes),
            max(box[2] for box in boxes), max(box[3] for box in boxes)]


def unique_in_order(values: Iterable[str]) -> List[str]:
    return list(dict.fromkeys(values))


def infer_split(language: str, image_name: str, index: int) -> str:
    if language == 'zh':
        return 'val' if '_val_' in image_name else 'train'
    # English SRFUND preserves the official FUNSD insertion order: 149 train,
    # followed by 50 test documents (named val here for SWIFT consistency).
    return 'train' if index < 149 else 'val'


def convert_language(instances: Dict, language: str, output_dir: Path,
                     source_image_root: Path | None, archive: zipfile.ZipFile | None,
                     extract_images: bool) -> Tuple[int, int]:
    output_image_dir = output_dir / 'images' / language
    if extract_images:
        output_image_dir.mkdir(parents=True, exist_ok=True)
    writers = {
        split: (output_dir / f'srfund_{language}_{split}.jsonl').open('w', encoding='utf-8')
        for split in ('train', 'val')
    }
    counts = {'train': 0, 'val': 0}
    try:
        for image_index, (image_name, entities) in enumerate(instances.items()):
            source_image_path = (source_image_root / language / image_name
                                 if source_image_root is not None else None)
            if source_image_path is not None:
                if not source_image_path.is_file():
                    raise FileNotFoundError(f'SRFUND image not found: {source_image_path}')
                image_bytes = source_image_path.read_bytes() if extract_images else None
                image_source = source_image_path
                image_path = output_image_dir / image_name if extract_images else source_image_path
            else:
                if archive is None:
                    raise RuntimeError('Neither an extracted image root nor dataset.zip is available.')
                if not extract_images:
                    raise ValueError('A zip-only source requires --extract-images.')
                image_bytes = archive.read(f'dataset/images/{language}/{image_name}')
                image_source = io.BytesIO(image_bytes)
                image_path = output_image_dir / image_name
            with Image.open(image_source) as image:
                width, height = image.size
            if extract_images and not image_path.exists():
                image_path.parent.mkdir(parents=True, exist_ok=True)
                assert image_bytes is not None
                image_path.write_bytes(image_bytes)
            entities_by_id = {int(entity['id']): entity for entity in entities}
            split = infer_split(language, image_name, image_index)
            grouped_questions = {}
            for question in entities:
                if question.get('label', '').lower() != 'question':
                    continue
                question_text = clean_entity_text(question.get('text', ''), question=True)
                answers, answer_policy = select_and_order_answers(linked_answers(question, entities_by_id))
                answer_pairs = [(answer, clean_entity_text(answer.get('text', ''))) for answer in answers]
                answer_pairs = [(answer, text) for answer, text in answer_pairs if text]
                if not question_text or not answer_pairs:
                    continue
                group = grouped_questions.setdefault(question_text, {
                    'question_ids': [], 'answer_pairs': [], 'policies': []})
                group['question_ids'].append(int(question['id']))
                group['answer_pairs'].extend(answer_pairs)
                group['policies'].append(answer_policy)

            for question_text, group in grouped_questions.items():
                # The prompt contains no question bbox, so duplicate field names
                # in one image cannot be supervised unambiguously.
                question_ids = unique_in_order(group['question_ids'])
                if len(question_ids) > 1:
                    continue
                pairs_by_id = {int(answer['id']): (answer, text)
                               for answer, text in group['answer_pairs']}
                answer_pairs = sorted(pairs_by_id.values(), key=lambda pair: (
                    union_entity_box(pair[0])[1], union_entity_box(pair[0])[0], int(pair[0]['id'])))
                answers = [answer for answer, _ in answer_pairs]
                answer_parts = unique_in_order(text for _, text in answer_pairs)
                answer_text = ' '.join(answer_parts)
                answer_bbox = normalize_bbox(union_answer_boxes(answers), width, height)
                answer_ids = [int(answer['id']) for answer in answers]
                answer_policy = group['policies'][0]
                sample_id = f'{language}:{image_name}:{question_ids[0]}'
                target = f'<ref>{answer_text}</ref><box>[[{",".join(map(str, answer_bbox))}]]</box>'
                instruction = ('请定位并回答字段：' if language == 'zh' else 'Locate and answer the field: ')
                row = {
                    'id': sample_id,
                    'images': [image_path.as_posix()],
                    'messages': [
                        {'role': 'user', 'content': '<image>\n' + instruction + question_text},
                        {'role': 'assistant', 'content': target},
                    ],
                    'language': language,
                    'split': split,
                    'question': question_text,
                    'answer': answer_text,
                    'answer_bbox': answer_bbox,
                    'answer_entity_ids': answer_ids,
                    'answer_count': len(answer_ids),
                    'question_entity_ids': question_ids,
                    'question_count': len(question_ids),
                    'answer_policy': answer_policy,
                    # Filled by run_ppocrv5.py. This path contains predictions only.
                    'ocr_cache': (output_dir / 'ocr_cache' / language / f'{image_name}.json').as_posix(),
                }
                writers[split].write(json.dumps(row, ensure_ascii=False) + '\n')
                counts[split] += 1
    finally:
        for writer in writers.values():
            writer.close()
    return counts['train'], counts['val']


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path('data/SRFUND/source'))
    parser.add_argument('--output-dir', type=Path,
                        default=Path('annotations/SRFUND'))
    parser.add_argument('--image-root', type=Path,
                        help='Existing image root containing zh/ and en/. Defaults to SOURCE/extracted/images.')
    parser.add_argument('--languages', nargs='+', choices=['zh', 'en'], default=['zh', 'en'])
    parser.add_argument('--extract-images', action=argparse.BooleanOptionalAction, default=False,
                        help='Copy images into OUTPUT_DIR/images. Disabled by default.')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    source_dir = args.source if args.source.is_dir() else None
    extracted_root = source_dir / 'extracted' if source_dir is not None else None
    source_image_root = args.image_root
    if source_image_root is None and extracted_root is not None:
        candidate = extracted_root / 'images'
        source_image_root = candidate if candidate.is_dir() else None
    archive_path = source_dir / 'dataset.zip' if source_dir is not None else args.source
    archive = zipfile.ZipFile(archive_path) if archive_path.is_file() else None
    try:
        for language in args.languages:
            annotation_path = (extracted_root / 'annotations' / 'instance' / f'{language}.json'
                               if extracted_root is not None else None)
            if annotation_path is not None and annotation_path.is_file():
                instances = json.loads(annotation_path.read_text(encoding='utf-8'))
            elif archive is not None:
                instances = json.loads(archive.read(f'dataset/instance_annotation/{language}.json'))
            else:
                raise FileNotFoundError(f'No instance annotation found for language={language}.')
            train_count, val_count = convert_language(
                instances, language, args.output_dir, source_image_root, archive, args.extract_images)
            print(f'{language}: train={train_count}, val={val_count}')
    finally:
        if archive is not None:
            archive.close()


if __name__ == '__main__':
    main()
