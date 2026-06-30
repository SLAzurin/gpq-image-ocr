from io import BytesIO
import sys
from typing import Dict, List, Tuple
import numpy
import pytesseract
import cv2
import string
import json
import os
from PIL import Image
from difflib import SequenceMatcher
from datetime import datetime
from enum import Enum
import unicodedata
import click
import base64
import font_match

# This program was entirely written by my friend qbkl
# I only added code optimizations

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        "C:/Program Files/Tesseract-OCR/tesseract.exe"
    )

class ComparisonTextType(Enum):
    ALUM = 1
    NUMS = 2


class ImageStyle(Enum):
    BIG = "big"
    SMALL = "small"


def splitImage(
    im: Image.Image, style: ImageStyle = ImageStyle.BIG
) -> Tuple[Image.Image, Image.Image]:
    if style == ImageStyle.BIG:
        # Full GPQ screenshot (~529x640): resize then crop name and score columns
        resized = im.resize((528, 642))
        im1 = resized.crop((45, 85, 120, 500))
        im2 = resized.crop((364, 85, 420, 500))
    else:
        # Pre-cropped table image (~447x413): crop name and GPQ score columns directly.
        # Name column ends at ~68px; keeping it tight prevents the adjacent class column
        # (Buccaneer, Demon Slayer, …) from bleeding into the name OCR read.
        im1 = im.crop((0, 0, 68, im.height))
        im2 = im.crop((305, 0, 415, im.height))
    return im1, im2


def readMembers(fileName: str) -> List[str]:
    with open(fileName + ".json", "r", encoding="utf8") as f:
        data: List[str] = json.loads(f.read())
    return data


_TEXT_TOL = 20
_WHITE_MIN = 255 - _TEXT_TOL          # #FFFFFF match: all channels >= this
_GRAY_TARGET = 179                     # #B3B3B3
_GRAY_LO = _GRAY_TARGET - _TEXT_TOL
_GRAY_HI = _GRAY_TARGET + _TEXT_TOL
_GRAY_SPREAD_MAX = 25                  # max-min channel diff to qualify as grey


def _binarize(pilImage: Image.Image) -> numpy.ndarray:
    """Force game-UI text colours to black and everything else to white.

    The game uses exactly two text colours:
      #FFFFFF (255,255,255) — highlighted / bold rows
      #B3B3B3 (179,179,179) — normal rows

    All other pixels (dark background, UI chrome, icons) become white.
    The result is a clean black-on-white image that gives Tesseract a high-
    contrast, noise-free signal before any scaling takes place.
    """
    arr = numpy.array(pilImage.convert("RGB"), dtype=numpy.uint8)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]

    is_white = (r >= _WHITE_MIN) & (g >= _WHITE_MIN) & (b >= _WHITE_MIN)

    spread = (
        numpy.maximum(numpy.maximum(r, g), b).astype(numpy.int32)
        - numpy.minimum(numpy.minimum(r, g), b).astype(numpy.int32)
    )
    is_gray = (
        (r >= _GRAY_LO) & (r <= _GRAY_HI)
        & (g >= _GRAY_LO) & (g <= _GRAY_HI)
        & (b >= _GRAY_LO) & (b <= _GRAY_HI)
        & (spread <= _GRAY_SPREAD_MAX)
    )

    out = numpy.full_like(arr, 255)       # white background
    out[is_white | is_gray] = 0           # text pixels → black
    return out


def _preprocess(
    pilImage: Image.Image, scale: int, close_strokes: bool = False
) -> numpy.ndarray:
    """Return a thresholded grayscale image suitable for Tesseract (dark text on white).

    Applies colour-based binarization first so that only the two game text
    colours survive, then scales and re-thresholds with Otsu.

    close_strokes: morphological closing fills small gaps inside digit strokes
    (e.g. the curved top of '5' that Tesseract may discard as noise).
    """
    binary = _binarize(pilImage)
    img = cv2.cvtColor(binary, cv2.COLOR_RGB2BGR)
    scaled = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(scaled, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    if close_strokes:
        # Closing fills small white gaps inside dark digit strokes.
        kernel = numpy.ones((2, 2), numpy.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    return thresh


def readImg(
    pilImage: Image.Image, textType: ComparisonTextType
) -> Dict[int, Dict[str, int]] | List[str]:
    match textType:
        case ComparisonTextType.ALUM:
            cfg = "-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàáâãäåéêëìíîïóôõöòøùúûüýÿÀÁÂÃÄÅÉÊËÌÍÎÏÓÔÕÖÒØÙÚÛÜÝŸ"
            scales = [4, 5]  # two passes for voting
            close_strokes = False
        case ComparisonTextType.NUMS:
            cfg = "-c tessedit_char_whitelist=0123456789"
            # Three passes for voting; closing fills broken strokes (e.g. leading '5')
            scales = [3, 4, 5]
            close_strokes = True
        case _:
            print("Invalid ComparisonTextType!")
            sys.exit(1)

    accuracyTable: Dict[int, Dict[str, int]] = {}

    for scale in scales:
        thresh = _preprocess(pilImage, scale, close_strokes=close_strokes)
        res: str = pytesseract.image_to_string(thresh, config=f"{cfg} --psm 6")
        # Split on newlines so each leaderboard row stays a single entry.
        # Taking only the FIRST whitespace-separated token per line discards
        # any adjacent-column bleed (e.g. "Ranul Dawr" → "Ranul",
        # "185118 1" → "185118") without losing the real value.
        lines = [ln.strip() for ln in res.split("\n") if ln.strip()]
        for i, line in enumerate(lines):
            token = line.split()[0]
            token = token.strip().translate(str.maketrans("", "", string.punctuation))
            if not token:
                continue
            accuracyTable.setdefault(i, {}).setdefault(token, 0)
            accuracyTable[i][token] += 1

    if textType == ComparisonTextType.NUMS:
        # Return the most-voted token for each row position
        return [max(accuracyTable[i], key=accuracyTable[i].get) for i in accuracyTable]
    return accuracyTable


def _fold(s: str) -> str:
    """Lowercase and strip all diacritical marks for accent-insensitive comparison.

    "Mïnäh" → "minah", "Kagètsu" → "kagetsu"
    """
    return "".join(
        c
        for c in unicodedata.normalize("NFD", s.lower())
        if unicodedata.category(c) != "Mn"
    )


def compNames(
    accuracyTable: Dict[int, Dict[str, int]], memberList: List[str]
) -> List[str]:
    res: List[str] = []
    for line in accuracyTable:
        occurence = 0
        isNewMember = True
        currentResult = ""
        currentTry = ""
        for ocrStr in accuracyTable[line]:
            if len(ocrStr) < 3:
                continue
            x = ocrStr.translate(str.maketrans("", "", string.punctuation))

            xf = _fold(x)
            xl = x.lower()

            # Tier 1 — exact case-insensitive prefix match (accents preserved).
            # Prioritised so that if OCR correctly captured an accent, the accented
            # member name wins over a plain-ASCII member with the same base letters.
            for m in memberList:
                if m.lower().startswith(xl):
                    currentResult = m
                    isNewMember = False
                    break
            if currentResult:
                break

            # Tier 2 — accent-folded prefix match (fallback when OCR dropped accents).
            # e.g. OCR reads "Minah" → folds to "minah" → matches member "Mïnäh".
            # Only reached when Tier 1 found nothing, so an exact-accent member always
            # beats a fold-only match.
            if len(x) >= 5:
                for m in memberList:
                    if _fold(m).startswith(xf):
                        currentResult = m
                        isNewMember = False
                        break
            if currentResult:
                break

            # Tier 3 — accent-folded fuzzy similarity, threshold 0.65
            FUZZY_THRESHOLD = 0.65
            compVal = FUZZY_THRESHOLD
            currName = ""
            for y in memberList:
                yf = _fold(y)
                if len(x) < 10:
                    actual = SequenceMatcher(None, xf, yf).ratio()
                    if actual > compVal:
                        compVal = actual
                        currName = y
                else:
                    trunc = SequenceMatcher(None, xf[:10], yf[:10]).ratio()
                    if trunc > compVal:
                        compVal = trunc
                        currName = y
            if accuracyTable[line][ocrStr] > occurence:
                if currName and compVal > FUZZY_THRESHOLD:
                    currentTry = currName
                    isNewMember = False
                else:
                    currentTry = ocrStr
                occurence = accuracyTable[line][ocrStr]
        if currentResult == "":
            currentResult = currentTry
        if len(currentResult) < 3:
            continue
        res.append(currentResult)
    return res


# ── Pixel template matching (lossless image path) ─────────────────────────────
# The game renders text in Arial 9pt (12px) with no antialiasing.  For lossless
# images we binarise to the game's two text colours and match each glyph against
# the generated font templates — far more accurate than OCR on this bitmap font.
# (Lossy video still uses Tesseract, since compression noise breaks pixel match.)

_FONT = None


def _get_font():
    global _FONT
    if _FONT is None:
        _FONT = font_match.load_font()
    return _FONT


def _binarize_bw(pilImage: Image.Image) -> numpy.ndarray:
    """Binarize to black text (0) on white (255), single channel."""
    rgb = _binarize(pilImage)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    return cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)[1]


# Minimum likeliness for a decoded name to be treated as an existing member.
# Below this we assume it's a new/unknown member and keep the literal decode.
NAME_MATCH_THRESHOLD = 0.7


def _norm(s: str) -> str:
    """Accent-fold and unify l/I (identical glyphs in this font)."""
    return _fold(s).replace("l", "i")


def _name_likeliness(a: str, b: str) -> float:
    """Confidence in [0,1] that decoded name `a` refers to member `b`.

    Both inputs are already accent/l-I folded.  Accounts for the two decode
    failure modes that should still resolve to a *known* member rather than a
    new entry:
      - truncation: the game clips wide names, so `a` is a clean prefix of `b`.
      - tail corruption: leading glyphs are correct, the last one or two are
        misread (e.g. "curseofrc" for "curseofyoshi" — they share "curseof").
    """
    if a == b:
        return 1.0
    if len(a) >= 3 and b.startswith(a):
        return 0.97
    lcp = 0
    for ca, cb in zip(a, b):
        if ca != cb:
            break
        lcp += 1
    # Fraction of the decode that agrees from the start; only trusted once a
    # solid run (>=4) matches, so unrelated short names don't slip through.
    prefix_score = lcp / len(a) if lcp >= 4 else 0.0
    ratio = SequenceMatcher(None, a, b).ratio()
    return max(ratio, prefix_score)


def reconcileName(dec: str, memberList: List[str]) -> str:
    """Resolve a decoded name to its best-matching member.

    Prefers fixing truncation/corruption against a known member; only when the
    best likeliness falls below NAME_MATCH_THRESHOLD is it treated as a new
    member and kept as the literal pixel decode (closest to the PNG).
    """
    if len(dec) < 2:
        return dec
    df = _norm(dec)
    best_member = ""
    best_conf = 0.0
    for m in memberList:
        conf = _name_likeliness(df, _norm(m))
        if conf > best_conf:
            best_conf, best_member = conf, m
    return best_member if best_conf >= NAME_MATCH_THRESHOLD else dec


def readNamesPixel(pilImage: Image.Image, memberList: List[str]) -> List[str]:
    bw = _binarize_bw(pilImage)
    decoded = font_match.decode_column(bw, _get_font())
    return [reconcileName(d.rstrip("."), memberList) for d in decoded]


def readScoresPixel(pilImage: Image.Image) -> List[str]:
    bw = _binarize_bw(pilImage)
    decoded = font_match.decode_column(bw, _get_font())
    return ["".join(c for c in d if c.isdigit()) for d in decoded]


def mergeScoresWithNames(
    gpq: List[str], names: List[str], currentState: Dict[str, int]
) -> Dict[str, int]:
    memberDict: Dict[str, int] = currentState
    for i in range(len(gpq)):
        try:
            score = int(gpq[i])
            if score > 0:
                memberDict[names[i]] = score
        except:
            break
    return memberDict


def videoToImages(path: str) -> List[Image.Image]:
    cap = cv2.VideoCapture(path)
    i = 0
    ret, frame_prev = cap.read()
    images: List[Image.Image] = [
        Image.fromarray(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB))
    ]
    i = 1
    while cap.isOpened():
        ret, frame_cur = cap.read()
        if ret == False:
            break
        diff = cv2.absdiff(frame_prev, frame_cur)
        mean_diff = diff.mean()
        if mean_diff > 3:
            images.append(Image.fromarray(cv2.cvtColor(frame_cur, cv2.COLOR_BGR2RGB)))
            frame_prev = frame_cur
        i += 1
    cap.release()
    cv2.destroyAllWindows()
    return images


@click.command()
@click.option("--subprocess", default=False, help="Number of greetings.")
@click.option("--video", default="", help="Use video file as input.")
@click.option(
    "--style",
    default="big",
    type=click.Choice(["big", "small"]),
    show_default=True,
    help="Image style: 'big' for full GPQ screenshots, 'small' for pre-cropped score table images.",
)
@click.option("--no-stdin", "no_stdin", is_flag=True, default=False, help="Skip the 'Press enter' prompt and exit immediately.")
def main(subprocess, video, style, no_stdin):
    """If this is a subprocess, expect json as stdin: { members: string[]; base64image: string } (without html data header for base64)"""
    """If there is a video path, expect json as stdin: { members: string[] }"""
    img_style = ImageStyle.BIG if style == "big" else ImageStyle.SMALL
    members: List[str] = []
    images: List[Image.Image] = []
    if not subprocess:
        print("Thank you for using gpq-image-ocr!\n")
        print("Made by:")
        print("qbkl (inuwater)")
        print("AzurinDayo (iMonoxian)\n")
        print("Other contributors:")
        print("YellowCello (BlueFlute)\n")
        print("Processing images...")
        members = readMembers("members")
        scoresDir = sorted(
            os.listdir(os.getcwd() + "/scores"),
            key=lambda f: int(os.path.splitext(f)[0])
            if os.path.splitext(f)[0].isdigit()
            else float("inf"),
        )
        for fname in scoresDir:
            if not fname.lower().endswith(".png"):
                continue
            images.append(Image.open(os.getcwd() + "/scores/" + fname))
    else:
        stdin = ""
        for line in sys.stdin:
            stdin += line.rstrip()
        stdinData = json.loads(stdin)
        members: List[str] = stdinData["members"]
        if not video == "":
            images = videoToImages(video)
        else:
            images = [Image.open(BytesIO(base64.b64decode(stdinData["base64image"])))]

    use_video = bool(video)
    memberDict: Dict[str, int] = {}
    for img in images:
        croppedNamesImage, croppedScoresImage = splitImage(
            img, ImageStyle.BIG if video else img_style
        )

        img.close()

        if use_video:
            # Lossy video: pixel matching is unreliable (compression noise), so
            # keep the Tesseract OCR + fuzzy-vote pipeline.
            readNameList = readImg(croppedNamesImage, ComparisonTextType.ALUM)
            if type(readNameList) is not dict:
                print(
                    "did not get Dict[int, Dict[str, int]] for readNameList, got ",
                    type(readNameList),
                )
                sys.exit(1)
            scores = readImg(croppedScoresImage, ComparisonTextType.NUMS)
            if type(scores) is not list:
                print("did not get List[str] for scores, got ", type(scores))
                sys.exit(1)
            actualNames = compNames(readNameList, members)
        else:
            # Lossless image: read the bitmap font directly via template matching.
            actualNames = readNamesPixel(croppedNamesImage, members)
            scores = readScoresPixel(croppedScoresImage)

        if len(actualNames) != len(scores):
            print(
                f"  Warning: members({len(actualNames)}) and scores({len(scores)}) count mismatch — "
                "padding shorter list with placeholders."
            )
            # Pad the shorter list so mergeScoresWithNames can still zip them
            while len(actualNames) < len(scores):
                actualNames.append(f"__unknown_{len(actualNames)}__")
            scores = scores[: len(actualNames)]
        mergeScoresWithNames(scores, actualNames, memberDict)

    if not subprocess:
        fName = "gpq_" + datetime.now().strftime("%m-%d-%Y") + ".json"
        with open(fName, "w", encoding="utf8") as f:
            json.dump(memberDict, f, ensure_ascii=False, indent=4)
        print("Done")
        print(f"The results are exported in {fName}")
        if not no_stdin:
            input("Press enter to close this window...")
    else:
        print(json.dumps(memberDict, ensure_ascii=False))


if __name__ == "__main__":
    main()
