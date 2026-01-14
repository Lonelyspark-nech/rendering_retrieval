#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import io
import re
import gc
import json
import argparse
import subprocess
import multiprocessing as mp
import shutil
from typing import Dict, Any, Tuple, Optional, Iterable, List

from tqdm import tqdm
from PIL import Image
Image.MAX_IMAGE_PIXELS = None

from pdf2image import convert_from_bytes

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT, TA_JUSTIFY
from reportlab.lib import colors
from xml.sax.saxutils import escape as xml_escape

import word2png_function as w2p

# Optional wheel (PDF first-page text extractor)
try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:
    _HAVE_FITZ = False


ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}
RE_CJK = re.compile(r"[\u4E00-\u9FFF]")


# ========== utils ==========
def count_lines_fast(path: str) -> int:
    env = os.environ.copy()
    env.setdefault("LC_ALL", "C")
    out = subprocess.check_output(["wc", "-l", path], text=True, env=env)
    return int(out.strip().split()[0])


def atomic_write_bytes(path: str, b: bytes) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        f.write(b)
    os.replace(tmp, path)


def atomic_save_png(img: Image.Image, path: str, compress_level: int = 1) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    img.save(tmp, format="PNG", compress_level=int(compress_level))
    os.replace(tmp, path)


def try_link_or_copy(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    tmp = dst + ".tmp"
    try:
        if os.path.exists(tmp):
            os.remove(tmp)
        os.link(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)


def _safe_filename(s: str) -> str:
    s = str(s)
    s = s.replace(os.sep, "_")
    s = s.replace("..", "_")
    s = re.sub(r"[^\w.\-]", "_", s)
    return s


def _clean_text_like_pipeline(text: str) -> str:
    # keep consistent with your earlier pipeline
    text = text.replace("\xad", "").replace("\u200b", "")
    text = text.replace("\r\n", "").replace("\n", "").replace("\r", "")
    return text


def _replace_multi_spaces_with_nbsp(s: str) -> str:
    return re.sub(r" {2,}", lambda m: "&nbsp;" * len(m.group()), s)


def _parse_alignment(v: Any) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        vv = v.strip().upper()
        return ALIGN_MAP.get(vv, TA_JUSTIFY)
    return TA_JUSTIFY


def _as_color(v: Any, default_hex: str) -> colors.Color:
    """
    config可能返回 str('#RRGGBB') 或 reportlab Color 对象。
    """
    if v is None:
        return colors.HexColor(default_hex)
    if isinstance(v, colors.Color):
        return v
    if isinstance(v, str):
        try:
            return colors.HexColor(v)
        except Exception:
            return colors.HexColor(default_hex)
    return colors.HexColor(default_hex)


def _pdf_first_page_text_pymupdf(pdf_bytes: bytes) -> str:
    """
    从生成的 PDF 第一页回读文本（同源对齐）。
    为了与你的 _clean_text_like_pipeline 对齐：把换行直接删除（不插入空格）。
    """
    if not _HAVE_FITZ:
        return ""
    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        try:
            if len(doc) == 0:
                return ""
            page0 = doc[0]
            t = page0.get_text("text") or ""
        finally:
            doc.close()
        # normalize like your pipeline: remove line breaks
        t = t.replace("\r\n", "\n").replace("\r", "\n").replace("\n", "")
        t = t.replace("\xa0", " ")
        return t
    except Exception:
        return ""


def _build_first_page_pdf_and_text(text_clean: str, cfg: Dict[str, Any], strict_font: bool) -> Tuple[bytes, str]:
    """
    高性能 & 同源对齐：
      - 用 Paragraph.split(avail_w, avail_h) 只截取“第一页能放下的段落对象”
      - 只 build 这一页（不会排版全文）
      - 文本不从 Paragraph/frags 拿，而是从生成的 PDF 第一页回读（同源）
    """
    page_size = cfg.get("page-size")
    if isinstance(page_size, str):
        page_size = tuple(map(float, page_size.split(",")))
    if page_size is None:
        page_size = (595.27, 841.89)

    margin_x = float(cfg.get("margin-x", 20))
    margin_y = float(cfg.get("margin-y", 20))

    font_path = cfg["font-path"]
    font_name = os.path.basename(font_path).split(".")[0]
    font_size = float(cfg.get("font-size", 9))
    line_height = float(cfg.get("line-height") or (font_size + 1))

    # colors
    page_bg_color = _as_color(cfg.get("page-bg-color"), "#FFFFFF")
    font_color = _as_color(cfg.get("font-color"), "#000000")
    para_bg_color = _as_color(cfg.get("para-bg-color"), "#FFFFFF")
    para_border_color = _as_color(cfg.get("para-border-color"), "#FFFFFF")

    first_line_indent = float(cfg.get("first-line-indent", 0))
    left_indent = float(cfg.get("left-indent", 0))
    right_indent = float(cfg.get("right-indent", 0))
    alignment = _parse_alignment(cfg.get("alignment", "JUSTIFY"))
    space_before = float(cfg.get("space-before", 0))
    space_after = float(cfg.get("space-after", 0))
    border_width = float(cfg.get("border-width", 0))
    border_padding = float(cfg.get("border-padding", 0))

    # Determinism: avoid silent fallback fonts
    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except Exception as e:
        if strict_font:
            raise RuntimeError(f"Failed to register font '{font_path}'. Err={e!r}")
        # allow fallback if not strict

    styles = getSampleStyleSheet()
    style = ParagraphStyle(
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
        wordWrap="CJK" if RE_CJK.search(text_clean) else None,
        leftIndent=left_indent,
        rightIndent=right_indent,
        alignment=alignment,
        spaceBefore=space_before,
        spaceAfter=space_after,
    )

    processed = _replace_multi_spaces_with_nbsp(xml_escape(text_clean))
    p_full = Paragraph(processed, style)

    avail_w = float(page_size[0]) - 2 * margin_x
    avail_h = float(page_size[1]) - 2 * margin_y

    # ⭐ 关键：只截断到“第一页内容”，避免全文排版
    parts = p_full.split(avail_w, avail_h)
    p_first = parts[0] if parts else p_full

    # Build 1-page PDF (only first page content)
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=page_size,
        leftMargin=margin_x,
        rightMargin=margin_x,
        topMargin=margin_y,
        bottomMargin=margin_y,
    )

    def paint_bg(c, d):
        c.saveState()
        c.setFillColor(page_bg_color)
        c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1)
        c.restoreState()

    doc.build([p_first], onFirstPage=paint_bg, onLaterPages=paint_bg)
    pdf_bytes = buf.getvalue()
    buf.close()

    # ✅ 唯一真相：从 PDF 第一页回读文本（同源）
    first_text = _pdf_first_page_text_pymupdf(pdf_bytes)

    # 温和兜底：不做“硬失败”，但避免无意义空文本
    if (not first_text or first_text.strip() == "") and text_clean.strip() != "":
        # 没装 fitz 或极少数提取失败：先不严苛，保底给一段
        first_text = text_clean[:4096]

    return pdf_bytes, first_text


def _pdf_first_page_to_png(pdf_bytes: bytes, cfg: Dict[str, Any], use_pdftocairo: bool) -> Image.Image:
    dpi = int(cfg.get("dpi", 72))
    horizontal_scale = float(cfg.get("horizontal-scale", 1.0))

    imgs = convert_from_bytes(
        pdf_bytes,
        dpi=dpi,
        first_page=1,
        last_page=1,
        thread_count=1,
        use_pdftocairo=bool(use_pdftocairo),
    )
    if not imgs:
        raise RuntimeError("convert_from_bytes returned empty images")
    img = imgs[0]

    if horizontal_scale != 1.0:
        w, h = img.size
        img = img.resize((int(w * horizontal_scale), h))

    return img


# ========== render plan ==========
def _parse_pagesize_key(k: str) -> Tuple[float, float]:
    kk = k.strip().lower().replace("x", ",")
    parts = [p.strip() for p in kk.split(",") if p.strip()]
    if len(parts) != 2:
        raise ValueError(f"Invalid page size key: {k!r}")
    return float(parts[0]), float(parts[1])


def load_render_plan(path: str) -> List[Tuple[Tuple[float, float], List[int]]]:
    """
    JSON must contain key 'page_sizes'
    Support:
      A) {"page_sizes": {"512,512":[48,72], "768x768":[60,72]}}
      B) {"page_sizes": [{"size":[512,512],"dpis":[48,72]}, {"size":"768x768","dpis":[60]}]}
    """
    with open(path, "r", encoding="utf-8") as f:
        plan = json.load(f)

    ps = plan.get("page_sizes")
    if ps is None:
        raise KeyError("Render plan JSON must contain key 'page_sizes'")

    out: List[Tuple[Tuple[float, float], List[int]]] = []
    if isinstance(ps, dict):
        for k, v in ps.items():
            size = _parse_pagesize_key(k)
            if not isinstance(v, list) or not v:
                raise ValueError(f"DPIs for {k!r} must be a non-empty list")
            dpis = sorted({int(x) for x in v})
            out.append((size, dpis))
        out.sort(key=lambda x: (x[0][0] * x[0][1], x[0][0], x[0][1]))
    elif isinstance(ps, list):
        for item in ps:
            if not isinstance(item, dict):
                raise ValueError("page_sizes list items must be objects")
            size_v = item.get("size")
            dpis_v = item.get("dpis")
            if size_v is None or dpis_v is None:
                raise ValueError(f"Each item must have 'size' and 'dpis': {item}")
            if isinstance(size_v, str):
                size = _parse_pagesize_key(size_v)
            else:
                if not (isinstance(size_v, (list, tuple)) and len(size_v) == 2):
                    raise ValueError(f"Invalid size: {size_v}")
                size = (float(size_v[0]), float(size_v[1]))
            if not isinstance(dpis_v, list) or not dpis_v:
                raise ValueError(f"Invalid dpis: {dpis_v}")
            dpis = sorted({int(x) for x in dpis_v})
            out.append((size, dpis))
    else:
        raise ValueError("'page_sizes' must be dict or list")

    return out


# ========== multiprocessing globals ==========
_G: Dict[str, Any] = {}


def _mp_init_layout(config_path: str, override: Dict[str, Any], layout_dir: str, strict_font: bool):
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    cfg = w2p.load_config(config_path)
    if override:
        cfg.update(override)

    _G["cfg"] = cfg
    _G["layout_dir"] = layout_dir
    _G["pdf_dir"] = os.path.join(layout_dir, "pdf_first_pages")
    _G["strict_font"] = bool(strict_font)

    os.makedirs(_G["pdf_dir"], exist_ok=True)


def _mp_init_raster(config_path: str, override: Dict[str, Any], layout_dir: str,
                    out_dir: str, png_compress_level: int, recover: bool, use_pdftocairo: bool):
    os.environ.setdefault("PYTHONHASHSEED", "0")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    cfg = w2p.load_config(config_path)
    if override:
        cfg.update(override)

    _G["cfg"] = cfg
    _G["layout_dir"] = layout_dir
    _G["pdf_dir"] = os.path.join(layout_dir, "pdf_first_pages")
    _G["out_dir"] = out_dir
    _G["img_dir"] = os.path.join(out_dir, "images")
    _G["png_compress_level"] = int(png_compress_level)
    _G["recover"] = bool(recover)
    _G["use_pdftocairo"] = bool(use_pdftocairo)

    os.makedirs(_G["img_dir"], exist_ok=True)


def _worker_layout(args: Tuple[int, str]) -> Optional[Tuple[int, str]]:
    """
    Input: (idx, json_line)
    Output: (idx, out_json_line) with schema order: doc_id, qid, text
    Also writes cached first-page PDF: <layout_dir>/pdf_first_pages/<doc_id>.pdf
    """
    try:
        idx, line = args
        obj = json.loads(line)

        doc_id = str(obj.get("doc_id", f"doc_{idx}"))
        qid = obj.get("qid", "")
        raw_text = obj.get("text", "")
        if not isinstance(qid, str):
            qid = str(qid)
        if not isinstance(raw_text, str):
            raw_text = str(raw_text)

        safe_id = _safe_filename(doc_id)
        pdf_path = os.path.join(_G["pdf_dir"], f"{safe_id}.pdf")

        text_clean = _clean_text_like_pipeline(raw_text)
        pdf_bytes, first_text = _build_first_page_pdf_and_text(
            text_clean=text_clean,
            cfg=_G["cfg"],
            strict_font=_G["strict_font"],
        )

        atomic_write_bytes(pdf_path, pdf_bytes)

        out_obj = {
            "doc_id": doc_id,
            "qid": qid,
            "text": first_text,
        }
        out_line = json.dumps(out_obj, ensure_ascii=False, separators=(",", ":"))
        return idx, out_line
    except Exception as e:
        err_path = os.path.join(_G["layout_dir"], "layout_errors.log")
        with open(err_path, "a", encoding="utf-8") as ef:
            ef.write(f"idx={args[0]}\t{repr(e)}\n")
        return None


def _worker_raster(args: Tuple[int, str]) -> Optional[int]:
    """
    Input: (idx, json_line) from layout corpus
    Output: idx
    """
    try:
        idx, line = args
        obj = json.loads(line)
        doc_id = str(obj["doc_id"])
        safe_id = _safe_filename(doc_id)

        pdf_path = os.path.join(_G["pdf_dir"], f"{safe_id}.pdf")
        img_path = os.path.join(_G["img_dir"], f"{safe_id}.png")

        if _G["recover"] and os.path.exists(img_path):
            return idx

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        img = _pdf_first_page_to_png(pdf_bytes, _G["cfg"], use_pdftocairo=_G["use_pdftocairo"])
        atomic_save_png(img, img_path, compress_level=_G["png_compress_level"])
        img.close()
        del pdf_bytes
        gc.collect()
        return idx
    except Exception as e:
        err_path = os.path.join(_G["out_dir"], "rasterize_errors.log")
        with open(err_path, "a", encoding="utf-8") as ef:
            ef.write(f"idx={args[0]}\t{repr(e)}\n")
        return None


def _iter_jsonl(path: str, limit: int = -1) -> Iterable[Tuple[int, str]]:
    """
    ⭐ 关键：limit 必须作用在“喂给 Pool 的迭代器”上，才能真正限制计算量。
    """
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit > 0 and i >= limit:
                break
            line = line.rstrip("\n")
            if not line:
                continue
            yield i, line


def layout_once_for_pagesize(
    corpus_path: str,
    layout_dir: str,
    config_path: str,
    override: Dict[str, Any],
    processes: int,
    mp_chunksize: int,
    recover: bool,
    limit: int,
    strict_font: bool,
) -> str:
    """
    Stage-1: layout once per (dataset, page_size)
      - output: <layout_dir>/corpus.jsonl (ordered)
      - cache:  <layout_dir>/pdf_first_pages/*.pdf
    """
    os.makedirs(layout_dir, exist_ok=True)
    os.makedirs(os.path.join(layout_dir, "pdf_first_pages"), exist_ok=True)

    out_corpus = os.path.join(layout_dir, "corpus.jsonl")
    if recover and os.path.exists(out_corpus):
        return out_corpus

    total = count_lines_fast(corpus_path)
    if limit > 0:
        total = min(total, limit)

    it = _iter_jsonl(corpus_path, limit=limit)

    with open(out_corpus, "w", encoding="utf-8") as fout:
        if processes <= 1:
            _mp_init_layout(config_path, override, layout_dir, strict_font)
            pbar = tqdm(total=total, desc=os.path.basename(layout_dir), dynamic_ncols=True, unit="doc")
            for item in it:
                r = _worker_layout(item)
                if r is not None:
                    _, out_line = r
                    fout.write(out_line + "\n")
                pbar.update(1)
            pbar.close()
        else:
            with mp.Pool(
                processes=processes,
                initializer=_mp_init_layout,
                initargs=(config_path, override, layout_dir, strict_font),
            ) as pool:
                pbar = tqdm(total=total, desc=os.path.basename(layout_dir), dynamic_ncols=True, unit="doc")
                for r in pool.imap(_worker_layout, it, chunksize=int(mp_chunksize)):
                    if r is not None:
                        _, out_line = r
                        fout.write(out_line + "\n")
                    pbar.update(1)
                pbar.close()

    return out_corpus


def rasterize_for_dpi(
    layout_corpus_path: str,
    layout_dir: str,
    out_dir: str,
    config_path: str,
    override: Dict[str, Any],
    processes: int,
    mp_chunksize: int,
    png_compress_level: int,
    recover: bool,
    limit: int,
    use_pdftocairo: bool,
) -> None:
    """
    Stage-2: rasterize per dpi from cached PDFs (no re-layout)
      - copy/link corpus.jsonl from layout dir
      - create images/*.png
    """
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)

    dst_corpus = os.path.join(out_dir, "corpus.jsonl")
    if not (recover and os.path.exists(dst_corpus)):
        try_link_or_copy(layout_corpus_path, dst_corpus)

    total = count_lines_fast(layout_corpus_path)
    if limit > 0:
        total = min(total, limit)

    it = _iter_jsonl(layout_corpus_path, limit=limit)

    if processes <= 1:
        _mp_init_raster(config_path, override, layout_dir, out_dir, png_compress_level, recover, use_pdftocairo)
        pbar = tqdm(total=total, desc=os.path.basename(out_dir), dynamic_ncols=True, unit="doc")
        for item in it:
            _worker_raster(item)
            pbar.update(1)
        pbar.close()
    else:
        with mp.Pool(
            processes=processes,
            initializer=_mp_init_raster,
            initargs=(config_path, override, layout_dir, out_dir, png_compress_level, recover, use_pdftocairo),
        ) as pool:
            pbar = tqdm(total=total, desc=os.path.basename(out_dir), dynamic_ncols=True, unit="doc")
            for _ in pool.imap(_worker_raster, it, chunksize=int(mp_chunksize)):
                pbar.update(1)
            pbar.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longembed_root", default="/data3/sunbo/ocr2/datasets/dwzhu/LongEmbed")
    ap.add_argument("--out_root", default="/data3/sunbo/ocr2/rendering_glyph/longembed_rendered_v4")
    ap.add_argument("--config_path", default="/data3/sunbo/ocr2/rendering_glyph/config_en.json")
    ap.add_argument("--render_plan_path", required=True)
    ap.add_argument("--datasets", nargs="*", default=["2wikimqa", "narrativeqa", "qmsum", "summ_screen_fd"])
    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--mp_chunksize", type=int, default=32)
    ap.add_argument("--png_compress_level", type=int, default=1)
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--strict_font", action="store_true")
    ap.add_argument("--use_pdftocairo", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    plan = load_render_plan(args.render_plan_path)

    for dataset in args.datasets:
        corpus_path = os.path.join(args.longembed_root, dataset, "corpus.jsonl")
        if not os.path.exists(corpus_path):
            print(f"[skip] corpus not found: {corpus_path}")
            continue

        print("\n" + "=" * 100)
        print(f"[DATASET] {dataset}")
        print("=" * 100)

        for (w, h), dpis in plan:
            ps_name = f"ps{int(w)}x{int(h)}"
            layout_dir = os.path.join(args.out_root, "layout_cache", ps_name, dataset)
            override_layout = {"page-size": (float(w), float(h))}
            print(f"\n[LAYOUT] dataset={dataset} page_size=({w},{h}) -> {layout_dir}")

            layout_corpus = layout_once_for_pagesize(
                corpus_path=corpus_path,
                layout_dir=layout_dir,
                config_path=args.config_path,
                override=override_layout,
                processes=args.processes,
                mp_chunksize=args.mp_chunksize,
                recover=args.recover,
                limit=args.limit,
                strict_font=args.strict_font,
            )

            for dpi in dpis:
                setting_name = f"{ps_name}_dpi{int(dpi)}"
                out_dir = os.path.join(args.out_root, setting_name, dataset)
                override_raster = {"page-size": (float(w), float(h)), "dpi": int(dpi)}
                print(f"[RASTERIZE] dataset={dataset} setting={setting_name} -> {out_dir}")

                rasterize_for_dpi(
                    layout_corpus_path=layout_corpus,
                    layout_dir=layout_dir,
                    out_dir=out_dir,
                    config_path=args.config_path,
                    override=override_raster,
                    processes=args.processes,
                    mp_chunksize=args.mp_chunksize,
                    png_compress_level=args.png_compress_level,
                    recover=args.recover,
                    limit=args.limit,
                    use_pdftocairo=args.use_pdftocairo,
                )

    print("\nAll done. Output root:", args.out_root)
    if not _HAVE_FITZ:
        print("[note] PyMuPDF (fitz) not found; text will fallback to text_clean[:4096]. Install pymupdf for best alignment.")


if __name__ == "__main__":
    main()
