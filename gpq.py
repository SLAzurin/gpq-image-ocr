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

# This program was entirely written by my friend qbkl
# I only added code optimizations

if os.name == "nt":
    pytesseract.pytesseract.tesseract_cmd = (
        "C:/Program Files/Tesseract-OCR/tesseract.exe"
    )


class ComparisonTextType(Enum):
    ALUM = 1
    NUMS = 2


def splitImage(f: str) -> Tuple[Image.Image, Image.Image]:
    im = Image.open("scores/" + f)
    resized = im.resize((528, 642))
    l1 = 45
    l2 = 364
    r1 = 120
    r2 = 420
    t = 85
    b = 500
    im1 = resized.crop((l1, t, r1, b))
    im2 = resized.crop((l2, t, r2, b))
    return im1, im2


def readMembers(fileName: str) -> List[str]:
    with open(fileName + ".json", "r", encoding="utf8") as f:
        data: List[str] = json.loads(f.read())
    return data


def readImg(
    pilImage: Image.Image, textType: ComparisonTextType
) -> Dict[int, Dict[str, int]] | List[str]:
    match textType:
        case ComparisonTextType.ALUM:
            cfg = (
                "-c tessedit_char_whitelist=0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZàáâãäåéêëìíîïóôõöòøùúûüýÿ"
                + "àáâãäåéêëìíîïóôõöòøùúûüýÿ".upper()
            )
            startRange = 3
            endRange = 7
        case ComparisonTextType.NUMS:
            cfg = "-c tessedit_char_whitelist=0123456789"
            startRange = 4
            endRange = 5
        case _:
            print("Invalid ComparisonTextType!")
            sys.exit(1)
    # Convert Image to cv2
    # pilImage.save("tmp.png")
    img = cv2.cvtColor(numpy.array(pilImage), cv2.COLOR_RGB2BGR)

    accuracyTable: Dict[
        int, Dict[str, int]
    ] = {}  # Dict[resListIndex, Dict[OCRName, occurenceCount]]

    for iteration in range(startRange, endRange):  # 3,4,5,6 resize
        scaled = cv2.resize(
            img, None, fx=iteration, fy=iteration, interpolation=cv2.INTER_CUBIC
        )
        HSV_img = cv2.cvtColor(scaled, cv2.COLOR_BGR2HSV)
        _, _, v = cv2.split(HSV_img)
        thresh = cv2.threshold(v, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        # cv2.imwrite(f"tmp{i}{textType}thresh.png", thresh)
        res: str = pytesseract.image_to_string(thresh, config=f"{cfg} --psm 6 digits")
        resList = res.split()
        for i in range(len(resList)):
            resList[i] = (
                resList[i].strip().translate(str.maketrans("", "", string.punctuation))
            )
            if i not in accuracyTable:
                accuracyTable[i] = {}
            if resList[i] not in accuracyTable[i]:
                accuracyTable[i][resList[i]] = 0
            accuracyTable[i][resList[i]] += 1
        # os.remove("tmp.png")
    if textType == ComparisonTextType.NUMS:
        r: List[str] = []
        for i in accuracyTable:
            r.append(list(accuracyTable[i].keys())[0])
        return r
    return accuracyTable


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

            # Instant break when it matches 100% of a member name in the defined list
            for m in memberList:
                if m.lower().startswith(x.lower()):
                    currentResult = m
                    isNewMember = False
                    break
            if currentResult != "":
                break

            # Compare with list of members below
            compVal = 0.7
            currName = ""
            for y in memberList:
                if len(x) < 10:
                    actual = SequenceMatcher(None, x, y).ratio()
                    if actual > compVal:
                        compVal = actual
                        currName = y
                else:
                    trunc = SequenceMatcher(None, x[:10], y[:10]).ratio()
                    if trunc > compVal:
                        compVal = trunc
                        currName = y
            if accuracyTable[line][ocrStr] > occurence:
                if currName != "" and compVal > 0.7:
                    currentTry = currName
                    isNewMember = False
                else:
                    currentTry = ocrStr
                occurence = accuracyTable[line][ocrStr]
        if currentResult == "":
            currentResult = currentTry
        if isNewMember:
            currentResult += "@NEWMEMBER"
        res.append(currentResult)
    return res


def createJson(gpq: List[str], names: List[str], memberDict: Dict[str, int]):
    for i in range(len(gpq)):
        try:
            score = int(gpq[i])
            if score > 0:
                memberDict[names[i]] = score
        except:
            break


def main():
    memberDict: Dict[str, int] = {}
    members = readMembers("members")
    lst = os.listdir(os.getcwd() + "/scores")
    for i in range(len(lst)):
        if not lst[i].lower().endswith(".png"):
            continue
        files = splitImage(lst[i])
        readNameList = readImg(files[0], ComparisonTextType.ALUM)
        if type(readNameList) is not dict:
            print(
                "did not get Dict[int, Dict[str, int]] for readNameList, got ",
                type(readNameList),
            )
            sys.exit(1)
        scores = readImg(files[1], ComparisonTextType.NUMS)
        if type(scores) is not list:
            print("did not get List[str] for scores, got ", type(scores))
            sys.exit(1)
        actualNames = compNames(readNameList, members)
        createJson(scores, actualNames, memberDict)
    fName = "gpq_" + datetime.now().strftime("%m-%d-%Y") + ".json"
    with open(fName, "w", encoding="utf8") as f:
        json.dump(memberDict, f, ensure_ascii=False, indent=4)
    return fName


if __name__ == "__main__":
    print("Thank you for using gpq-image-ocr!\n")
    print("Made by:")
    print("qbkl (inuwater)")
    print("AzurinDayo (iMonoxian)\n")
    print("Other contributors:")
    print("YellowCello (BlueFlute)\n")
    print("Processing images...")
    resultsFName = main()
    print("Done")
    print(f"The results are exported in {resultsFName}")
    input("Press enter to close this window...")
