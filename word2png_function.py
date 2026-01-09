#!/usr/bin/env python3
from PIL import Image
Image.MAX_IMAGE_PIXELS = None
import io
import os
import json
import numpy as np
import gc
from pdf2image import pdfinfo_from_bytes, convert_from_bytes
import re
from multiprocessing import Pool
from tqdm import tqdm
from xml.sax.saxutils import escape
import shutil

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.pagesizes import A4
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors

# Alignment mapping
ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}

# Global variables for multiprocessing
GLOBAL_CONFIG = None
OUTPUT_DIR = None
recover = False


def load_config(config_path):
    """Load configuration file"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Convert colors
    if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
        config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
    if 'font-color' in config and isinstance(config['font-color'], str):
        config['font-color'] = colors.HexColor(config['font-color'])
    if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
        config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
    if 'para-border-color' in config and isinstance(config['para-border-color'], str):
        config['para-border-color'] = colors.HexColor(config['para-border-color'])
    
    # Convert alignment
    if 'alignment' in config and isinstance(config['alignment'], str):
        config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
    
    # Convert page size
    if 'page-size' in config and isinstance(config['page-size'], str):
        config['page-size'] = tuple(map(float, config['page-size'].split(',')))
    
    return config


def text_to_images(
    text,
    output_dir,
    config_path=None,
    config_dict=None,
    unique_id=None,
    # --- page limiting / truncation output ---
    max_pages=None,                  # e.g. 1 => force <=1 page via truncation
    return_truncated_text=False,      # True => return (image_paths, truncated_processed_text)
    # --- output path control ---
    out_pattern: str = "{output_dir}/{unique_id}/page_{page:03d}.png",
    # --- pre-truncation acceleration ---
    pretruncate_chars: int = None,    # if set, pre-cut processed_text to this many chars * safety before binary search
    pretruncate_safety: float = 1.10, # safety multiplier for pretruncate (>=1.0)
):
    """
    Convert text to images (Glyph-like: ReportLab -> PDF -> pdf2image -> PNG).

    - Keeps original behavior by default: returns List[str] of all pages rendered from full text.
    - If max_pages is not None:
        * Finds the longest prefix of processed_text that renders to <= max_pages pages (binary search).
        * Rasterizes only the first max_pages pages.
        * If return_truncated_text=True, also returns that processed_text prefix.
    - Supports customizable save paths via out_pattern.
    - Supports pretruncate_chars to reduce binary-search cost in batch rendering.

    Args:
        text: input text (raw).
        output_dir: root output dir (used in out_pattern formatting).
        config_path/config_dict: same as original.
        unique_id: used in output path formatting and default directory structure.
        max_pages: None or int.
        return_truncated_text: bool.
        out_pattern: format string. Available fields: output_dir, unique_id, page.
        pretruncate_chars: int or None. Applied to processed_text BEFORE page-fitting search.
        pretruncate_safety: float, multiplier on pretruncate_chars.

    Returns:
        - if return_truncated_text=False: List[str]
        - else: (List[str], str)
          where str is truncated_processed_text (processed/escaped string prefix actually rendered)
    """
    # Load configuration (same logic as your wheel)
    if config_dict is None:
        if config_path is None:
            raise ValueError("Must provide either config_path or config_dict")
        config = load_config(config_path)
    else:
        config = config_dict.copy()
        # Convert special fields in config
        if 'page-bg-color' in config and isinstance(config['page-bg-color'], str):
            config['page-bg-color'] = colors.HexColor(config['page-bg-color'])
        if 'font-color' in config and isinstance(config['font-color'], str):
            config['font-color'] = colors.HexColor(config['font-color'])
        if 'para-bg-color' in config and isinstance(config['para-bg-color'], str):
            config['para-bg-color'] = colors.HexColor(config['para-bg-color'])
        if 'para-border-color' in config and isinstance(config['para-border-color'], str):
            config['para-border-color'] = colors.HexColor(config['para-border-color'])
        if 'alignment' in config and isinstance(config['alignment'], str):
            config['alignment'] = ALIGN_MAP.get(config['alignment'], TA_JUSTIFY)
        if 'page-size' in config and isinstance(config['page-size'], str):
            config['page-size'] = tuple(map(float, config['page-size'].split(',')))

    # Generate unique ID
    if unique_id is None:
        import hashlib
        unique_id = hashlib.md5(text.encode()).hexdigest()[:16]

    # Extract configuration parameters
    page_size = config.get('page-size', A4)
    margin_x = config.get('margin-x', 20)
    margin_y = config.get('margin-y', 20)
    font_path = config.get('font-path')
    assert font_path, "Must provide font-path"

    font_name = os.path.basename(font_path).split('.')[0]
    font_size = config.get('font-size', 9)
    line_height = config.get('line-height') or (font_size + 1)

    page_bg_color = config.get('page-bg-color', colors.HexColor('#FFFFFF'))
    font_color = config.get('font-color', colors.HexColor('#000000'))
    para_bg_color = config.get('para-bg-color', colors.HexColor('#FFFFFF'))
    para_border_color = config.get('para-border-color', colors.HexColor('#FFFFFF'))

    first_line_indent = config.get('first-line-indent', 0)
    left_indent = config.get('left-indent', 0)
    right_indent = config.get('right-indent', 0)
    alignment = config.get('alignment', TA_JUSTIFY)
    space_before = config.get('space-before', 0)
    space_after = config.get('space-after', 0)
    border_width = config.get('border-width', 0)
    border_padding = config.get('border-padding', 0)

    horizontal_scale = config.get('horizontal-scale', 1.0)
    dpi = config.get('dpi', 72)
    auto_crop_last_page = config.get('auto-crop-last-page', False)
    auto_crop_width = config.get('auto-crop-width', False)
    newline_markup = config.get('newline-markup', '<br/>')

    # Register font
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except Exception:
        pass  # already registered

    # Create paragraph style
    styles = getSampleStyleSheet()
    RE_CJK = re.compile(r'[\u4E00-\u9FFF]')

    custom = ParagraphStyle(
        name="Custom",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=font_size,
        leading=line_height,
        textColor=font_color,
        backColor=para_bg_color,
        borderColor=para_border_color,
        borderWidth=border_width,
        borderPadding=border_padding,
        firstLineIndent=first_line_indent,
        wordWrap="CJK" if RE_CJK.search(text) else None,
        leftIndent=left_indent,
        rightIndent=right_indent,
        alignment=alignment,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )

    # Process text (same as your current version: remove newlines only; preserve spaces)
    def replace_spaces(s):
        return re.sub(r' {2,}', lambda m: '&nbsp;' * len(m.group()), s)

    text = text.replace('\xad', '').replace('\u200b', '')
    text = text.replace('\r\n', '').replace('\n', '').replace('\r', '')
    processed_text = replace_spaces(escape(text))

    # Optional pre-truncation to speed up (applies to processed_text)
    if pretruncate_chars is not None:
        try:
            cap = int(float(pretruncate_chars) * float(pretruncate_safety))
        except Exception:
            cap = int(pretruncate_chars)
        if cap > 0 and len(processed_text) > cap:
            processed_text = processed_text[:cap]

    # Build PDF bytes helper (keeps background painting behavior)
    def build_pdf_bytes_from_processed(proc_txt: str) -> bytes:
        buf_local = io.BytesIO()
        doc_local = SimpleDocTemplate(
            buf_local,
            pagesize=page_size,
            leftMargin=margin_x,
            rightMargin=margin_x,
            topMargin=margin_y,
            bottomMargin=margin_y,
        )

        story_local = []
        turns = 30
        tmp_parts = [proc_txt]  # keep "no line split" behavior
        for i in range(0, len(tmp_parts), turns):
            tmp_text = newline_markup.join(tmp_parts[i:i + turns])
            story_local.append(Paragraph(tmp_text, custom))

        def _paint_bg(c, d):
            c.saveState()
            c.setFillColor(page_bg_color)
            c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1)
            c.restoreState()

        doc_local.build(story_local, onFirstPage=_paint_bg, onLaterPages=_paint_bg)
        pdfb = buf_local.getvalue()
        buf_local.close()
        return pdfb

    # If max_pages set, binary-search longest prefix that fits
    truncated_processed_text = processed_text
    if max_pages is not None:
        max_pages_i = int(max_pages)
        pdf_try = build_pdf_bytes_from_processed(processed_text)
        if int(pdfinfo_from_bytes(pdf_try)["Pages"]) > max_pages_i:
            lo, hi = 1, len(processed_text)
            best = 1
            while lo <= hi:
                mid = (lo + hi) // 2
                cand = processed_text[:mid]
                pdfb = build_pdf_bytes_from_processed(cand)
                if int(pdfinfo_from_bytes(pdfb)["Pages"]) <= max_pages_i:
                    best = mid
                    lo = mid + 1
                else:
                    hi = mid - 1
            truncated_processed_text = processed_text[:best]
            pdf_bytes = build_pdf_bytes_from_processed(truncated_processed_text)
        else:
            truncated_processed_text = processed_text
            pdf_bytes = pdf_try
    else:
        pdf_bytes = build_pdf_bytes_from_processed(processed_text)

    # Determine number of pages to rasterize
    info = pdfinfo_from_bytes(pdf_bytes)
    total = int(info["Pages"])
    if max_pages is not None:
        total = min(total, int(max_pages))

    batch = 20
    image_paths = []

    for start in range(1, total + 1, batch):
        end = min(start + batch - 1, total)
        images = convert_from_bytes(pdf_bytes, dpi=dpi, first_page=start, last_page=end)

        for offset, img in enumerate(images, start=start):
            w, h = img.size

            # Horizontal scaling
            if horizontal_scale != 1.0:
                img = img.resize((int(w * horizontal_scale), h))

            # Adaptive cropping (same as original wheel)
            if auto_crop_width or (auto_crop_last_page and offset == total):
                gray = np.array(img.convert("L"))
                bg_gray = np.median(gray[:2, :2])
                tolerance = 5
                mask = np.abs(gray - bg_gray) > tolerance

                if auto_crop_width:
                    cols = np.where(mask.any(axis=0))[0]
                    if cols.size:
                        rightmost_col = cols[-1] + 1
                        right = min(img.width, rightmost_col + margin_x)
                        img = img.crop((0, 0, right, img.height))

                if auto_crop_last_page and offset == total:
                    rows = np.where(mask.any(axis=1))[0]
                    if rows.size:
                        last_row = rows[-1]
                        lower = min(img.height, last_row + margin_y)
                        img = img.crop((0, 0, img.width, lower))

            # Save path controlled by out_pattern
            out_path = out_pattern.format(output_dir=output_dir, unique_id=unique_id, page=offset)
            os.makedirs(os.path.dirname(out_path), exist_ok=True)
            img.save(out_path, 'PNG')
            image_paths.append(os.path.abspath(out_path))
            img.close()

        images.clear()
        del images

    del pdf_bytes
    gc.collect()

    if return_truncated_text:
        return image_paths, truncated_processed_text
    return image_paths



def process_one(item):
    """Process single item - for batch processing"""
    global GLOBAL_CONFIG, OUTPUT_DIR, recover
    
    _id = item.get('unique_id')
    assert _id
    
    # Check recovery mode
    if recover and os.path.exists(os.path.join(OUTPUT_DIR, _id)):
        item['image_paths'] = []
        return item
    
    # Parse configuration
    item_config = item.get('config', {}) or {}
    config = {**GLOBAL_CONFIG, **item_config}
    
    # Process special fields in item config
    if 'page-size' in item_config and isinstance(item_config['page-size'], str):
        config['page-size'] = tuple(map(float, item_config['page-size'].split(',')))
    if 'page-bg-color' in item_config and isinstance(item_config['page-bg-color'], str):
        config['page-bg-color'] = colors.HexColor(item_config['page-bg-color'])
    if 'font-color' in item_config and isinstance(item_config['font-color'], str):
        config['font-color'] = colors.HexColor(item_config['font-color'])
    if 'para-bg-color' in item_config and isinstance(item_config['para-bg-color'], str):
        config['para-bg-color'] = colors.HexColor(item_config['para-bg-color'])
    if 'para-border-color' in item_config and isinstance(item_config['para-border-color'], str):
        config['para-border-color'] = colors.HexColor(item_config['para-border-color'])
    if 'alignment' in item_config and isinstance(item_config['alignment'], str):
        config['alignment'] = ALIGN_MAP.get(item_config['alignment'], TA_JUSTIFY)
    
    # Get text
    text = item.get('context', '')
    assert text
    
    # Call inference function
    image_paths = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR,
        config_dict=config,
        unique_id=_id
    )
    
    item['image_paths'] = image_paths
    return item


def batch_process_to_images(json_path, output_dir, output_jsonl_path, 
                            config_path, processes=16, is_recover=False, batch_size=100):
    """Batch process JSON data to generate images"""
    global GLOBAL_CONFIG, OUTPUT_DIR, recover
    
    # Set global variables
    GLOBAL_CONFIG = load_config(config_path)
    OUTPUT_DIR = output_dir
    recover = is_recover
    
    print(f"Loaded config from: {config_path}")
    
    # Prepare output directory
    if not recover:
        if os.path.isdir(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir, exist_ok=True)
        if os.path.exists(output_jsonl_path):
            os.remove(output_jsonl_path)
    
    # Read data
    with open(json_path, 'r', encoding='utf-8') as f:
        data_to_process = json.load(f)
    
    # Get already processed IDs
    processed_ids = set()
    if recover and os.path.exists(output_jsonl_path):
        with open(output_jsonl_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    processed_ids.add(item.get('unique_id'))
                except:
                    continue
        print(f"Found {len(processed_ids)} already processed items")
    
    # Filter processed items
    data_to_process = [item for item in data_to_process 
                      if item.get('unique_id') not in processed_ids]
    print(f"Remaining items to process: {len(data_to_process)}")
    
    if not data_to_process:
        print("All items processed")
        return
    
    # Parallel processing
    batch_buffer = []
    
    with Pool(processes=processes) as pool:
        for result_item in tqdm(pool.imap_unordered(process_one, data_to_process, chunksize=1), 
                               total=len(data_to_process)):
            if result_item:
                batch_buffer.append(result_item)
                _id = result_item.get('unique_id', 'UNKNOWN')
                count = len(result_item.get('image_paths', []))
                tqdm.write(f"{_id}: generated {count} pages")
                
                # Batch write
                if len(batch_buffer) >= batch_size:
                    with open(output_jsonl_path, 'a', encoding='utf-8') as f:
                        for item in batch_buffer:
                            f.write(json.dumps(item, ensure_ascii=False) + '\n')
                    batch_buffer = []
    
    # Write remaining items
    if batch_buffer:
        with open(output_jsonl_path, 'a', encoding='utf-8') as f:
            for item in batch_buffer:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
    
    print("Processing complete")


if __name__ == '__main__':
    # Example 1: Single text inference
    CONFIG_PATH = '../config/config.json'
    text = "This is a test text\nSecond line of text\nThird line of text"
    OUTPUT_DIR = './output_images'
    images = text_to_images(
        text=text,
        output_dir=OUTPUT_DIR,
        config_path=CONFIG_PATH,
        unique_id='test_001'
    )
    print(f"Generated {len(images)} images:")
    for img in images:
        print(f"  {img}")
    
    # Example 2: Batch processing
    # CONFIG_PATH = '../config/config.json'
    # JSON_PATH = '../evaluation/mrcr/data/processed_2needle_0-128k.json'
    # OUTPUT_JSONL_PATH = '../evaluation/mrcr/data/processed_2needle_0-128k.jsonl'
    # OUTPUT_DIR = '../evaluation/mrcr/rendered_images'
    
    # batch_process_to_images(
    #     json_path=JSON_PATH,
    #     output_dir=OUTPUT_DIR,
    #     output_jsonl_path=OUTPUT_JSONL_PATH,
    #     config_path=CONFIG_PATH,
    #     processes=16,
    #     is_recover=True,
    #     batch_size=100
    # )

