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

# -------------------------
# Optional dependency
# -------------------------
try:
    import fitz  # PyMuPDF
    _HAVE_FITZ = True
except Exception:
    _HAVE_FITZ = False


# =========================
# Constants / helpers
# =========================
ALIGN_MAP = {
    "LEFT": TA_LEFT,
    "CENTER": TA_CENTER,
    "RIGHT": TA_RIGHT,
    "JUSTIFY": TA_JUSTIFY,
}
RE_CJK = re.compile(r"[\u4E00-\u9FFF]")


def _parse_alignment(v: Any) -> int:
    if isinstance(v, int):
        return v
    if isinstance(v, str):
        return ALIGN_MAP.get(v.strip().upper(), TA_JUSTIFY)
    return TA_JUSTIFY


def _safe_filename(s: str) -> str:
    s = str(s)
    s = s.replace(os.sep, "_").replace("..", "_")
    return re.sub(r"[^\w.\-]", "_", s)


def _clean_text_like_pipeline(text: str) -> str:
    text = text.replace("\xad", "").replace("\u200b", "")
    text = text.replace("\r\n", "").replace("\n", "").replace("\r", "")
    return text


def _replace_multi_spaces_with_nbsp(s: str) -> str:
    return re.sub(r" {2,}", lambda m: "&nbsp;" * len(m.group()), s)


def count_lines_fast(path: str) -> int:
    out = subprocess.check_output(["wc", "-l", path], text=True)
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
        os.link(src, tmp)
        os.replace(tmp, dst)
    except Exception:
        shutil.copyfile(src, tmp)
        os.replace(tmp, dst)


# =========================
# PDF → Text (page 1)
# =========================
def _pdf_first_page_text_pymupdf(pdf_bytes: bytes, doc_id: str) -> str:
    if not _HAVE_FITZ:
        raise RuntimeError("PyMuPDF (fitz) is required")

    try:
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        if len(doc) == 0:
            raise RuntimeError("PDF has zero pages")

        page0 = doc[0]
        t = page0.get_text("text") or ""
        doc.close()

        # 与 pipeline 对齐：删除物理换行
        t = t.replace("\r\n", "\n").replace("\r", "\n")
        t = t.replace("\n", "")
        t = t.replace("\xa0", " ")

        return t
    except Exception as e:
        raise RuntimeError(f"[DocID={doc_id}] PDF text extraction failed: {e}")


# =========================
# Build PDF + extract text
# =========================
def _build_first_page_pdf_and_text(
    text_clean: str,
    cfg: Dict[str, Any],
    strict_font: bool,
    doc_id: str,
) -> Tuple[bytes, str]:

    def get_color(key, default):
        v = cfg.get(key, default)
        if isinstance(v, str):
            return colors.HexColor(v)
        return v

    page_size = cfg.get("page-size", (595.27, 841.89))
    if isinstance(page_size, str):
        page_size = tuple(map(float, page_size.split(",")))

    margin_x = float(cfg.get("margin-x", 20))
    margin_y = float(cfg.get("margin-y", 20))

    font_path = cfg["font-path"]
    font_name = os.path.basename(font_path).split(".")[0]

    try:
        pdfmetrics.registerFont(TTFont(font_name, font_path))
    except Exception as e:
        if strict_font:
            raise RuntimeError(f"Font register failed: {e}")

    styles = getSampleStyleSheet()
    style = ParagraphStyle(
        name="Custom",
        parent=styles["Normal"],
        fontName=font_name,
        fontSize=float(cfg.get("font-size", 9)),
        leading=float(cfg.get("line-height") or 10),
        textColor=get_color("font-color", "#000000"),
        backColor=get_color("para-bg-color", "#FFFFFF"),
        borderColor=get_color("para-border-color", "#FFFFFF"),
        borderWidth=float(cfg.get("border-width", 0)),
        borderPadding=float(cfg.get("border-padding", 0)),
        firstLineIndent=float(cfg.get("first-line-indent", 0)),
        leftIndent=float(cfg.get("left-indent", 0)),
        rightIndent=float(cfg.get("right-indent", 0)),
        spaceBefore=float(cfg.get("space-before", 0)),
        spaceAfter=float(cfg.get("space-after", 0)),
        alignment=_parse_alignment(cfg.get("alignment", "JUSTIFY")),
        wordWrap="CJK" if RE_CJK.search(text_clean) else None,
    )

    processed = _replace_multi_spaces_with_nbsp(xml_escape(text_clean))
    p_full = Paragraph(processed, style)

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
        bg = get_color("page-bg-color", "#FFFFFF")
        c.saveState()
        c.setFillColor(bg)
        c.rect(0, 0, page_size[0], page_size[1], stroke=0, fill=1)
        c.restoreState()

    # 核心：全文 build，自然分页
    doc.build([p_full], onFirstPage=paint_bg, onLaterPages=paint_bg)
    pdf_bytes = buf.getvalue()
    buf.close()

    first_text = _pdf_first_page_text_pymupdf(pdf_bytes, doc_id)

    # 硬校验：防止静默脏数据
    if text_clean.strip() and not first_text.strip():
        raise RuntimeError(f"[DocID={doc_id}] non-empty input but empty extracted text")

    return pdf_bytes, first_text


# =========================
# PDF → PNG
# =========================
def _pdf_first_page_to_png(pdf_bytes: bytes, cfg: Dict[str, Any], use_pdftocairo: bool) -> Image.Image:
    dpi = int(cfg.get("dpi", 72))
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
    return imgs[0]


# =========================
# Multiprocessing globals
# =========================
_G: Dict[str, Any] = {}


def _mp_init_layout(config_path, override, layout_dir, strict_font):
    cfg = w2p.load_config(config_path)
    if override:
        cfg.update(override)
    _G["cfg"] = cfg
    _G["layout_dir"] = layout_dir
    _G["pdf_dir"] = os.path.join(layout_dir, "pdf_first_pages")
    _G["strict_font"] = strict_font
    os.makedirs(_G["pdf_dir"], exist_ok=True)


def _mp_init_raster(config_path, override, layout_dir, out_dir, png_compress_level, recover, use_pdftocairo):
    cfg = w2p.load_config(config_path)
    if override:
        cfg.update(override)
    _G["cfg"] = cfg
    _G["layout_dir"] = layout_dir
    _G["pdf_dir"] = os.path.join(layout_dir, "pdf_first_pages")
    _G["out_dir"] = out_dir
    _G["img_dir"] = os.path.join(out_dir, "images")
    _G["png_compress_level"] = png_compress_level
    _G["recover"] = recover
    _G["use_pdftocairo"] = use_pdftocairo
    os.makedirs(_G["img_dir"], exist_ok=True)


# =========================
# Workers
# =========================
def _worker_layout(args):
    try:
        idx, line = args
        obj = json.loads(line)

        doc_id = str(obj.get("doc_id", f"doc_{idx}"))
        qid = str(obj.get("qid", ""))
        raw_text = str(obj.get("text", ""))

        safe_id = _safe_filename(doc_id)
        pdf_path = os.path.join(_G["pdf_dir"], f"{safe_id}.pdf")

        text_clean = _clean_text_like_pipeline(raw_text)

        pdf_bytes, first_text = _build_first_page_pdf_and_text(
            text_clean=text_clean,
            cfg=_G["cfg"],
            strict_font=_G["strict_font"],
            doc_id=doc_id,
        )

        atomic_write_bytes(pdf_path, pdf_bytes)

        out = {
            "doc_id": doc_id,
            "qid": qid,
            "text": first_text,
        }
        return idx, json.dumps(out, ensure_ascii=False, separators=(",", ":"))
    except Exception as e:
        with open(os.path.join(_G["layout_dir"], "layout_errors.log"), "a") as f:
            f.write(f"{args[0]}\t{repr(e)}\n")
        return None


def _worker_raster(args):
    try:
        idx, line = args
        obj = json.loads(line)
        doc_id = obj["doc_id"]
        safe_id = _safe_filename(doc_id)

        pdf_path = os.path.join(_G["pdf_dir"], f"{safe_id}.pdf")
        img_path = os.path.join(_G["img_dir"], f"{safe_id}.png")

        if _G["recover"] and os.path.exists(img_path):
            return idx

        with open(pdf_path, "rb") as f:
            pdf_bytes = f.read()

        img = _pdf_first_page_to_png(pdf_bytes, _G["cfg"], _G["use_pdftocairo"])
        atomic_save_png(img, img_path, _G["png_compress_level"])
        img.close()
        gc.collect()
        return idx
    except Exception as e:
        with open(os.path.join(_G["out_dir"], "rasterize_errors.log"), "a") as f:
            f.write(f"{args[0]}\t{repr(e)}\n")
        return None


# =========================
# Stage functions
# =========================
def layout_once_for_pagesize(corpus_path, layout_dir, config_path, override,
                             processes, mp_chunksize, recover, limit, strict_font):
    os.makedirs(layout_dir, exist_ok=True)
    out_corpus = os.path.join(layout_dir, "corpus.jsonl")
    if recover and os.path.exists(out_corpus):
        return out_corpus

    total = count_lines_fast(corpus_path)
    if limit > 0:
        total = min(total, limit)

    with open(out_corpus, "w", encoding="utf-8") as fout:
        with mp.Pool(processes, initializer=_mp_init_layout,
                     initargs=(config_path, override, layout_dir, strict_font)) as pool:
            for r in tqdm(pool.imap(_worker_layout, enumerate(open(corpus_path)), mp_chunksize),
                          total=total, desc=os.path.basename(layout_dir)):
                if r:
                    fout.write(r[1] + "\n")
    return out_corpus


def rasterize_for_dpi(layout_corpus_path, layout_dir, out_dir, config_path, override,
                      processes, mp_chunksize, png_compress_level, recover, limit, use_pdftocairo):
    os.makedirs(out_dir, exist_ok=True)
    try_link_or_copy(layout_corpus_path, os.path.join(out_dir, "corpus.jsonl"))

    total = count_lines_fast(layout_corpus_path)
    if limit > 0:
        total = min(total, limit)

    with mp.Pool(processes, initializer=_mp_init_raster,
                 initargs=(config_path, override, layout_dir, out_dir,
                           png_compress_level, recover, use_pdftocairo)) as pool:
        for _ in tqdm(pool.imap(_worker_raster, enumerate(open(layout_corpus_path)), mp_chunksize),
                      total=total, desc=os.path.basename(out_dir)):
            pass


# =========================
# Main
# =========================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--longembed_root", required=True)
    ap.add_argument("--out_root", required=True)
    ap.add_argument("--config_path", required=True)
    ap.add_argument("--render_plan_path", required=True)
    ap.add_argument("--datasets", nargs="+", required=True)
    ap.add_argument("--processes", type=int, default=16)
    ap.add_argument("--mp_chunksize", type=int, default=32)
    ap.add_argument("--png_compress_level", type=int, default=1)
    ap.add_argument("--recover", action="store_true")
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--strict_font", action="store_true")
    ap.add_argument("--use_pdftocairo", action="store_true")
    args = ap.parse_args()

    with open(args.render_plan_path, "r") as f:
        plan = json.load(f)["page_sizes"]

    for dataset in args.datasets:
        corpus_path = os.path.join(args.longembed_root, dataset, "corpus.jsonl")
        if not os.path.exists(corpus_path):
            continue

        for ps, dpis in plan.items():
            w, h = map(float, ps.replace("x", ",").split(","))
            layout_dir = os.path.join(args.out_root, "layout_cache", f"ps{int(w)}x{int(h)}", dataset)

            layout_corpus = layout_once_for_pagesize(
                corpus_path, layout_dir, args.config_path,
                {"page-size": (w, h)},
                args.processes, args.mp_chunksize,
                args.recover, args.limit, args.strict_font
            )

            for dpi in dpis:
                out_dir = os.path.join(args.out_root, f"ps{int(w)}x{int(h)}_dpi{dpi}", dataset)
                rasterize_for_dpi(
                    layout_corpus, layout_dir, out_dir, args.config_path,
                    {"page-size": (w, h), "dpi": dpi},
                    args.processes, args.mp_chunksize,
                    args.png_compress_level, args.recover,
                    args.limit, args.use_pdftocairo
                )


if __name__ == "__main__":
    main()
