"""
font_match.py

Bitmap-font glyph recognition by template matching, replacing Tesseract OCR
for the fixed-font GPQ leaderboard.

The game renders text in a fixed bitmap font using exactly two colours, so
after binarisation each glyph is one of a small set of known bitmaps.  We:

1. Load the per-character templates built by build_font.py (native 1x).
2. Detect text-row bands in a column crop via dark-pixel Y-projection.
3. Decode each row left-to-right: at the current x, find the template that
   best matches starting there (sliding vertically to handle ascenders /
   descenders / x-height), emit its character, and advance by its width.

Because templates are exact bitmaps, the correct glyph at the correct position
yields a (near) zero pixel-difference, which disambiguates touching and
proportional glyphs far more reliably than OCR.
"""

import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

FONT_DIR = "font"


# ── Font loading ──────────────────────────────────────────────────────────────

def load_font(font_dir: str = FONT_DIR) -> Dict[str, List[np.ndarray]]:
    """Load all glyph templates as binary (0/255) bitmaps keyed by character."""
    font: Dict[str, List[np.ndarray]] = {}
    if not os.path.isdir(font_dir):
        raise FileNotFoundError(f"Font directory '{font_dir}' not found.")
    for d in sorted(os.listdir(font_dir)):
        if not d.startswith("U"):
            continue
        try:
            ch = chr(int(d[1:], 16))
        except ValueError:
            continue
        tmpls: List[np.ndarray] = []
        ddir = os.path.join(font_dir, d)
        for f in os.listdir(ddir):
            if not f.endswith(".png"):
                continue
            img = cv2.imread(os.path.join(ddir, f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            _, b = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY)
            tmpls.append(b)
        if tmpls:
            font[ch] = tmpls
    return font


# ── Row detection ─────────────────────────────────────────────────────────────

def detect_rows(col_bin: np.ndarray) -> List[Tuple[int, int]]:
    """Return (y0, y1) bands for each text row in a binary column crop.

    Bands separated by <= 3 px are merged so a diacritic mark stays attached to
    its row.
    """
    dark = (col_bin < 128).sum(axis=1)
    bands: List[List[int]] = []
    in_b = False
    y0 = 0
    for y, d in enumerate(dark):
        if not in_b and d > 0:
            in_b, y0 = True, y
        elif in_b and d == 0:
            in_b = False
            bands.append([y0, y])
    if in_b:
        bands.append([y0, len(dark)])
    merged: List[List[int]] = []
    for b in bands:
        if merged and b[0] - merged[-1][1] <= 3:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    return [(b[0], b[1]) for b in merged]


# ── Glyph matching ────────────────────────────────────────────────────────────

def _best_glyph_at(
    row: np.ndarray, x: int, font: Dict[str, List[np.ndarray]]
) -> Optional[Tuple[str, int, float]]:
    """Find the best-matching template starting at column x.

    Returns (char, width, normalised_distance) or None.  Vertically slides each
    template within the row band and takes the minimum pixel-difference.  Ties
    on distance are broken toward the WIDEST glyph, which prevents a narrow
    template (e.g. 'i' / 'l') from spuriously matching the left edge of a wider
    glyph (e.g. 'n').
    """
    H, W = row.shape
    best: Optional[Tuple[float, int, str]] = None  # (dist_norm, -width, char)
    for ch, tmpls in font.items():
        for T in tmpls:
            hT, wT = T.shape
            if x + wT > W or hT > H:
                continue
            window = row[:, x:x + wT]  # full-height H x wT slice
            ink_total = int(np.count_nonzero(window < 128))
            best_d = None
            for y in range(0, H - hT + 1):
                # Place T at vertical offset y; everything outside is white.
                # Distance counts mismatches inside the template footprint PLUS
                # any ink outside it — so a short glyph (e.g. 'v') cannot match
                # the top of a taller glyph (e.g. 'y') and ignore the descender.
                band = window[y:y + hT, :]
                d_in = int(np.count_nonzero(band != T))
                ink_in_band = int(np.count_nonzero(band < 128))
                d_out = ink_total - ink_in_band  # ink above/below the placement
                d = d_in + d_out
                if best_d is None or d < best_d:
                    best_d = d
                    if best_d == 0:
                        break
            if best_d is None:
                continue
            dn = best_d / (H * wT)
            key = (dn, -wT, ch)
            if best is None or key < best:
                best = key
    if best is None:
        return None
    dn, neg_w, ch = best
    return ch, -neg_w, dn


def match_row(
    row: np.ndarray,
    font: Dict[str, List[np.ndarray]],
    tol: float = 0.12,
    max_skip: int = 4,
    gap_stop: int = 8,
) -> str:
    """Decode the first "word" of a binary text row (black text on white).

    tol      : max normalised pixel-difference to accept a glyph match.
    max_skip : consecutive unmatched ink columns to skip (e.g. a ',' separator
               inside a score, or minor noise) before giving up.
    gap_stop : once at least one glyph is emitted, a blank run this wide ends
               the field — this discards adjacent-column bleed (e.g. the
               '1,000' clears column intruding on a score crop), mirroring the
               old "first whitespace token" behaviour.  Intra-number digit gaps
               are <=4px while inter-column gaps are tens of px, so 8 is safe.
    """
    H, W = row.shape
    x = 0
    out: List[str] = []
    skipped = 0
    blank_run = 0

    def col_ink(c: int) -> bool:
        return bool(np.any(row[:, c] < 128))

    while x < W:
        if not col_ink(x):
            x += 1
            blank_run += 1
            skipped = 0
            if out and blank_run >= gap_stop:
                break
            continue
        blank_run = 0
        match = _best_glyph_at(row, x, font)
        if match is not None and match[2] <= tol:
            ch, wT, _ = match
            out.append(ch)
            x += wT
            skipped = 0
        else:
            # Unmatched ink (separator like ',' or noise): skip a few columns.
            x += 1
            skipped += 1
            if skipped > max_skip:
                break
    return "".join(out)


# ── Column decode ─────────────────────────────────────────────────────────────

def decode_column(
    col_bin: np.ndarray,
    font: Dict[str, List[np.ndarray]],
    tol: float = 0.12,
) -> List[str]:
    """Detect rows in a binary column crop and decode each to a string."""
    results: List[str] = []
    for y0, y1 in detect_rows(col_bin):
        results.append(match_row(col_bin[y0:y1, :], font, tol=tol))
    return results
