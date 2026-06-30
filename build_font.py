"""
build_font.py
Extracts per-character glyph templates from scores_panda/ images to build a
custom font library for template-matching OCR.
Strategy
--------
1. Detect text-row bands in each panda image's name column by projecting
   dark-pixel density onto the Y axis.
2. OCR each row band (multi-scale, parallel) and pick the most-voted token.
3. Identify the member via fold-normalised exact / prefix / fuzzy match
   against members.json.
4. Segment the name row into glyphs using vertical valley finding on a
   horizontally-eroded copy (so flush characters get a separating valley),
   then crop the matching x-ranges from the CLEAN image for crisp templates.
5. Accept a row only when glyph count == visible character count (accounting
   for trailing truncation dots), then save each glyph to
   font/U{codepoint:04X}/{index:03d}.png.
Incremental & idempotent
------------------------
Templates are deduplicated by pixel-content hash, so the run can be repeated
as new panda images arrive without wiping font/ and without creating
duplicates.  Manually-added PNGs (for characters that never appear in member
names) are preserved.
After each run the script prints which required characters still lack a
template — split into "could extract from more images" vs "needs a manual
PNG" — and writes font_preview.png for visual QA.
Usage
-----
    python build_font.py                       # (or 'extract') from scores_panda/
    python build_font.py ingest IMG TEXT       # add glyphs from a manual image
    python build_font.py ingest-dir FOLDER     # add glyphs from one-char PNGs
    python build_font.py generate              # (re)build font/ from Arial 9pt
    python build_font.py report                # coverage report + preview only
The `ingest` mode slices a single image of consecutive glyphs and files each
under the corresponding character of TEXT (whitespace ignored).  Use it to
supply the accents/digits that never appear in member names, e.g.:
    python build_font.py ingest accents.png "áâãåéêëq05678"
`ingest-dir` instead reads a folder of single-glyph PNGs whose filename stem is
the character (e.g. ä.png), bypassing segmentation entirely.  `generate`
recreates the whole library from the game's font (Arial 9pt / 12px, no
antialiasing) — the fastest way to get a complete, pixel-clean set.
"""
import argparse
import hashlib
import json
import os
import shutil
import string
import sys
from concurrent.futures import ThreadPoolExecutor
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import pytesseract
from PIL import Image, ImageDraw, ImageFont

import gpq
import font_match

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        "C:/Program Files/Tesseract-OCR/tesseract.exe"
    )

# ── Configuration ─────────────────────────────────────────────────────────────
FONT_DIR = "font"
PREVIEW_PATH = "font_preview.png"
PANDA_DIR = "scores_panda"
MEMBERS = "members"  # gpq.readMembers appends .json

ARIAL_PATH = "C:/Windows/Fonts/arial.ttf" if os.name == "nt" else "arial.ttf"
ARIAL_PX = 12  # 9pt at 96dpi — the game's rendered size

DIGITS = "0123456789"
ASCII_LOWER = "abcdefghijklmnopqrstuvwxyz"
ASCII_UPPER = ASCII_LOWER.upper()
ACCENTS = "àáâãäåéêëìíîïóôõöòøùúûüýÿ"
ACCENTS_UPPER = ACCENTS.upper()
# Full target alphabet, de-duplicated while preserving order.
REQUIRED = list(
    dict.fromkeys(DIGITS + ASCII_LOWER + ASCII_UPPER + ACCENTS + ACCENTS_UPPER)
)
WHITELIST = DIGITS + ASCII_LOWER + ASCII_UPPER + ACCENTS + ACCENTS_UPPER


# ── Low-level helpers ─────────────────────────────────────────────────────────

def _imread_unicode(path: str, flags: int = cv2.IMREAD_GRAYSCALE):
    """cv2.imread that tolerates non-ASCII paths (e.g. ä.png) on Windows."""
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if data.size == 0:
        return None
    return cv2.imdecode(data, flags)


def _tdir(char: str) -> str:
    return os.path.join(FONT_DIR, f"U{ord(char):04X}")


def _to_binary(gray: np.ndarray) -> np.ndarray:
    """Threshold a grayscale image to 0/255 with text as black (0)."""
    return cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]


def _tight_crop(bin_img: np.ndarray) -> Optional[np.ndarray]:
    """Crop a binary (text=0) image to its ink bounding box; None if blank."""
    ink = bin_img < 128
    if not ink.any():
        return None
    ys, xs = np.where(ink)
    crop = bin_img[ys.min(): ys.max() + 1, xs.min(): xs.max() + 1]
    return _to_binary(crop)


def _glyph_hash(tmpl: np.ndarray) -> str:
    """Content hash of a template for deduplication (shape + pixels)."""
    h = hashlib.md5()
    h.update(repr(tmpl.shape).encode())
    h.update(tmpl.tobytes())
    return h.hexdigest()


# ── Template store (incremental, idempotent) ──────────────────────────────────

class _Saver:
    """Writes glyph templates to font/, skipping pixel-identical duplicates.

    Indices continue from whatever already exists on disk, and previously saved
    (including manually-added) templates are preserved across runs.
    """

    def __init__(self) -> None:
        self._idx: Dict[str, int] = {}
        self._hashes: Dict[str, Set[str]] = {}
        self.saved = 0
        self._load_existing()

    def reset(self) -> None:
        self._idx.clear()
        self._hashes.clear()
        self.saved = 0

    def _load_existing(self) -> None:
        if not os.path.isdir(FONT_DIR):
            return
        for d in os.listdir(FONT_DIR):
            if not d.startswith("U"):
                continue
            try:
                char = chr(int(d[1:], 16))
            except ValueError:
                continue
            dirp = os.path.join(FONT_DIR, d)
            if not os.path.isdir(dirp):
                continue
            idxs: List[int] = []
            hs: Set[str] = set()
            for f in os.listdir(dirp):
                if not f.endswith(".png"):
                    continue
                stem = os.path.splitext(f)[0]
                if stem.isdigit():
                    idxs.append(int(stem))
                img = cv2.imread(os.path.join(dirp, f), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    hs.add(_glyph_hash(_to_binary(img)))
            self._idx[char] = max(idxs) + 1 if idxs else 0
            self._hashes[char] = hs

    def save(self, char: str, glyph: np.ndarray) -> bool:
        """Tight-crop, binarise and store a glyph. Returns False if duplicate."""
        cropped = _tight_crop(glyph)
        if cropped is None:
            return False
        digest = _glyph_hash(cropped)
        seen = self._hashes.setdefault(char, set())
        if digest in seen:
            return False
        idx = self._idx.get(char, 0)
        os.makedirs(_tdir(char), exist_ok=True)
        cv2.imwrite(os.path.join(_tdir(char), f"{idx:03d}.png"), cropped)
        self._idx[char] = idx + 1
        seen.add(digest)
        self.saved += 1
        return True


# ── Segmentation ──────────────────────────────────────────────────────────────

def _erode_sep(thresh: np.ndarray) -> np.ndarray:
    """Thin glyphs horizontally so touching characters gain a white valley.

    Text is black (0) on white (255), so growing the white background with a
    horizontal kernel erodes the glyphs from the sides without closing the
    vertical gaps that separate neighbours.
    """
    return cv2.dilate(thresh, np.ones((1, 3), np.uint8))


def _segment_ranges(bin_img: np.ndarray) -> List[Tuple[int, int]]:
    """Return (x0, x1) column runs that contain ink (text=0)."""
    ink = (bin_img < 128).any(axis=0)
    ranges: List[Tuple[int, int]] = []
    in_run = False
    x0 = 0
    for x, v in enumerate(ink):
        if v and not in_run:
            in_run, x0 = True, x
        elif not v and in_run:
            in_run = False
            ranges.append((x0, x))
    if in_run:
        ranges.append((x0, len(ink)))
    return ranges


def _segment_glyphs(row: np.ndarray) -> List[np.ndarray]:
    """Split a 1x text row into glyph crops via 4x eroded valley detection."""
    scaled = cv2.resize(row, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    ranges = _segment_ranges(_erode_sep(scaled))
    glyphs: List[np.ndarray] = []
    for a, b in ranges:
        # Map the 4x range back to 1x and crop from the CLEAN row.
        x0, x1 = a // 4, -(-b // 4)
        if x1 > x0:
            glyphs.append(row[:, x0:x1])
    return glyphs


# ── Member matching (for extract OCR) ─────────────────────────────────────────

def _match_member(token: str, members: List[str]) -> Optional[str]:
    """Resolve an OCR token to a member name via fold-normalised matching."""
    tok = token.strip().translate(str.maketrans("", "", string.punctuation))
    if len(tok) < 3:
        return None
    tf = gpq._fold(tok)
    for m in members:
        if gpq._fold(m).startswith(tf):
            return m
    best: Optional[str] = None
    best_r = 0.7
    for m in members:
        r = SequenceMatcher(None, tf, gpq._fold(m)).ratio()
        if r > best_r:
            best_r, best = r, m
    return best


# ── Extract from scores_panda/ ─────────────────────────────────────────────────

def _name_rows() -> List[Tuple[str, np.ndarray]]:
    """Return (source_path, 1x binary row image) for every detected name row."""
    rows: List[Tuple[str, np.ndarray]] = []
    paths = sorted(
        os.path.join(PANDA_DIR, f)
        for f in os.listdir(PANDA_DIR)
        if f.lower().endswith(".png")
    ) if os.path.isdir(PANDA_DIR) else []
    for path in paths:
        try:
            pil = Image.open(path)
        except OSError:
            print(f"  [skip] cannot read {path}")
            continue
        names, _ = gpq.splitImage(pil, gpq.ImageStyle.SMALL)
        clean = gpq._binarize(names)
        bw = _to_binary(cv2.cvtColor(clean, cv2.COLOR_RGB2GRAY))
        for y0, y1 in font_match.detect_rows(bw):
            rows.append((path, bw[y0:y1, :].copy()))
    return rows


def _ocr_row(row: np.ndarray) -> str:
    """Multi-scale OCR of a single name row; returns the most-voted token."""
    votes: Dict[str, int] = {}
    for scale in (4, 5):
        up = cv2.resize(row, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        cfg = f"-c tessedit_char_whitelist={WHITELIST} --psm 7"
        txt = pytesseract.image_to_string(up, config=cfg).strip()
        parts = txt.split()
        if not parts:
            continue
        tok = parts[0]
        votes[tok] = votes.get(tok, 0) + 1
    return max(votes, key=votes.get) if votes else ""


def _extract(saver: _Saver, members: List[str]) -> None:
    jobs = _name_rows()
    panda_count = len({p for p, _ in jobs})
    print(f"OCR-ing {len(jobs)} rows across {panda_count} images...")
    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as ex:
        tokens = list(ex.map(lambda j: _ocr_row(j[1]), jobs))

    matched_rows = 0
    for (_, row), token in zip(jobs, tokens):
        name = _match_member(token, members)
        if not name:
            continue
        glyphs = _segment_glyphs(row)
        # Only learn from rows whose segmentation matches the full member name;
        # truncated rows (glyph count < name length) are skipped.
        if len(glyphs) != len(name):
            continue
        for ch, glyph in zip(name, glyphs):
            saver.save(ch, glyph)
        matched_rows += 1
    print(f"Matched {matched_rows}/{len(jobs)} rows  |  saved {saver.saved} new templates")


# ── Manual ingestion ───────────────────────────────────────────────────────────

def _ingest_manual(image_path: str, text: str, saver: _Saver) -> None:
    """Slice one image of consecutive glyphs into per-character templates."""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        sys.exit("No characters given to ingest.")
    img = _imread_unicode(image_path)
    if img is None:
        sys.exit(f"Cannot read image: {image_path}")
    thresh = _to_binary(img)

    # Single glyph: skip segmentation/erosion (which can destroy thin 1x strokes).
    if len(chars) == 1:
        if saver.save(chars[0], thresh):
            print(f"  saved U+{ord(chars[0]):04X} '{chars[0]}'")
        else:
            print(f"  skipped U+{ord(chars[0]):04X} '{chars[0]}' (duplicate)")
        print(f"Ingested {saver.saved} new glyph(s) from {image_path}")
        return

    scaled = cv2.resize(thresh, None, fx=4, fy=4, interpolation=cv2.INTER_NEAREST)
    ranges = _segment_ranges(_erode_sep(scaled))
    glyphs = [thresh[:, a // 4: -(-b // 4)] for a, b in ranges]

    if len(glyphs) != len(chars):
        dbg = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR)
        for a, b in ranges:
            cv2.rectangle(dbg, (a // 4, 0), (-(-b // 4) - 1, dbg.shape[0] - 1), (0, 0, 255), 1)
        cv2.imwrite("_ingest_debug.png", dbg)
        sys.exit(
            f"Segmentation mismatch: found {len(glyphs)} glyphs but text has "
            f"{len(chars)} characters.\n"
            f"  glyphs split at x-ranges: {ranges}\n"
            f"  Saved _ingest_debug.png showing the splits (red boxes).\n"
            f"  Tip: ensure clear gaps between characters, single row, no border."
        )

    for char, glyph in zip(chars, glyphs):
        if saver.save(char, glyph):
            print(f"  saved U+{ord(char):04X} '{char}'")
        else:
            print(f"  skipped U+{ord(char):04X} '{char}' (duplicate)")
    print(f"Ingested {saver.saved} new glyph(s) from {image_path}")


def _ingest_dir(folder: str, saver: _Saver) -> None:
    """Ingest a folder of single-glyph PNGs named by their character (ä.png)."""
    for p in sorted(Path(folder).glob("*.png")):
        if len(p.stem) != 1:
            print(f"  [skip] {p.name}: filename stem must be exactly one character")
            continue
        char = p.stem
        img = _imread_unicode(str(p))
        if img is None:
            print(f"  [skip] {p.name}: cannot read")
            continue
        crop = _tight_crop(_to_binary(img))
        if crop is None:
            print(f"  [skip] {p.name}: blank after thresholding")
            continue
        if saver.save(char, crop):
            print(f"  saved U+{ord(char):04X} '{char}'  ({crop.shape[0]}x{crop.shape[1]}) from {p.name}")
        else:
            print(f"  dup   U+{ord(char):04X} '{char}' from {p.name}")
    print(f"Ingested {saver.saved} new glyph(s) from {folder}/")


# ── Arial generation ───────────────────────────────────────────────────────────

def _render_arial(ch: str, px: int = ARIAL_PX, font_path: str = ARIAL_PATH) -> Optional[np.ndarray]:
    """Render a single glyph in Arial with NO antialiasing, text=0 on white.

    Draws onto a generously oversized canvas at a fixed inset so the glyph is
    never clipped, with fontmode '1' to disable antialiasing, then tight-crops
    to the ink.  This reproduces the game's 9pt/12px no-AA bitmaps exactly.
    """
    font = ImageFont.truetype(font_path, px)
    canvas = px * 4
    img = Image.new("L", (canvas, canvas), 255)
    draw = ImageDraw.Draw(img)
    draw.fontmode = "1"  # disable antialiasing
    draw.text((px, px), ch, font=font, fill=0)
    return _tight_crop(_to_binary(np.array(img, dtype=np.uint8)))


def _generate_arial(saver: _Saver, font_path: str = ARIAL_PATH, px: int = ARIAL_PX) -> None:
    """Wipe font/ and regenerate every required glyph from Arial (no-AA)."""
    if not os.path.exists(font_path):
        sys.exit(f"Font file not found: {font_path}")
    if os.path.isdir(FONT_DIR):
        shutil.rmtree(FONT_DIR)
    os.makedirs(FONT_DIR, exist_ok=True)
    saver.reset()

    missing: List[str] = []
    for ch in REQUIRED:
        glyph = _render_arial(ch, px, font_path)
        if glyph is None:
            missing.append(ch)
            continue
        saver.save(ch, glyph)
    print(f"Generated {saver.saved} glyph(s) from Arial {px}px (no-AA).")
    if missing:
        print("  Arial produced no ink for: " + " ".join(f"U+{ord(c):04X}" for c in missing))


# ── Reporting & preview ────────────────────────────────────────────────────────

def _report_coverage(members: List[str]) -> None:
    have = {c for c in REQUIRED if os.path.isdir(_tdir(c)) and os.listdir(_tdir(c))}
    missing = [c for c in REQUIRED if c not in have]
    print(f"\nCoverage: {len(have)}/{len(REQUIRED)} characters have >=1 template")
    if not missing:
        print("\nAll required characters covered.")
        return

    in_names: Dict[str, List[str]] = {}
    need_png: List[str] = []
    for c in missing:
        users = [m for m in members if c in m]
        if users:
            in_names[c] = users
        else:
            need_png.append(c)

    if in_names:
        print("\nMissing but PRESENT in member names (more/clearer images may capture them):")
        for c, users in in_names.items():
            print(f"  U+{ord(c):04X}  '{c}'  — in: {', '.join(users)}")
    if need_png:
        print("\nMissing and NOT in any member name — supply a PNG sample for these:")
        print("  " + "  ".join(f"U+{ord(c):04X} '{c}'" for c in need_png))


def _build_preview(scale: int = 6, cols: int = 16, pad: int = 4, label_h: int = 14) -> None:
    cells: List[Tuple[str, np.ndarray]] = []
    for c in REQUIRED:
        d = _tdir(c)
        if not os.path.isdir(d):
            continue
        files = sorted(f for f in os.listdir(d) if f.endswith(".png"))
        if not files:
            continue
        g = cv2.imread(os.path.join(d, files[0]), cv2.IMREAD_GRAYSCALE)
        if g is not None:
            cells.append((c, g))
    if not cells:
        return

    cell_w = max(g.shape[1] for _, g in cells) * scale + pad * 2
    cell_h = max(g.shape[0] for _, g in cells) * scale + pad * 2 + label_h
    rows = (len(cells) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), (40, 40, 40))
    draw = ImageDraw.Draw(canvas)
    for i, (c, g) in enumerate(cells):
        r, cc = divmod(i, cols)
        ox, oy = cc * cell_w, r * cell_h
        draw.text((ox + 2, oy + 1), f"{ord(c):04X}", fill=(170, 170, 170))
        up = cv2.resize(g, (g.shape[1] * scale, g.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
        glyph_img = Image.fromarray(cv2.cvtColor(up, cv2.COLOR_GRAY2RGB))
        canvas.paste(glyph_img, (ox + pad, oy + label_h + pad))
    canvas.save(PREVIEW_PATH)


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Build the glyph template font library.")
    sub = parser.add_subparsers(dest="cmd")
    sub.add_parser("extract", help="extract glyphs from scores_panda/ (default)")
    p_ing = sub.add_parser("ingest", help="add glyphs from one multi-char image")
    p_ing.add_argument("image")
    p_ing.add_argument("text")
    p_dir = sub.add_parser("ingest-dir", help="add glyphs from one-char PNGs in a folder")
    p_dir.add_argument("folder")
    p_gen = sub.add_parser("generate", help="(re)build font/ from Arial (no-AA)")
    p_gen.add_argument("--font", default=ARIAL_PATH)
    p_gen.add_argument("--px", type=int, default=ARIAL_PX)
    sub.add_parser("report", help="coverage report + preview only")
    args = parser.parse_args()

    members = gpq.readMembers(MEMBERS)
    saver = _Saver()

    if args.cmd in (None, "extract"):
        _extract(saver, members)
    elif args.cmd == "ingest":
        _ingest_manual(args.image, args.text, saver)
    elif args.cmd == "ingest-dir":
        _ingest_dir(args.folder, saver)
    elif args.cmd == "generate":
        _generate_arial(saver, font_path=args.font, px=args.px)

    _report_coverage(members)
    _build_preview()
    print(f"\nWrote {PREVIEW_PATH} for visual verification.")


if __name__ == "__main__":
    main()
