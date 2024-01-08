import pytesseract
import cv2
import string
import json
import os
from PIL import Image
from difflib import SequenceMatcher
from datetime import datetime

# This program was entirely written by my friend qbkl
# I only added code optimizations

if os.name == 'nt':
    pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"

def splitImage(f):
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


def readMembers(fileName):
    with open(fileName+'.json', "r") as f:
        data = json.loads(f.read())
    return data


def readImg(pilImage):
    pilImage.save("tmp.png")
    img = cv2.imread("tmp.png")
    scaled = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    res = pytesseract.image_to_string(scaled)
    resList = res.split()
    for i in range(len(resList)):
        resList[i] = resList[i].strip()
        resList[i] = resList[i].translate(str.maketrans("", "", string.punctuation))
    os.remove("tmp.png")
    return resList


def compNames(names, memberList):
    res = []
    for x in names:
        x = x.translate(str.maketrans("", "", string.punctuation))
        found = False
        for y in memberList:
            if SequenceMatcher(None, x, y).ratio() >= 0.6:
                # print(x, "<->", y, SequenceMatcher(None, x, y).ratio())
                found = True
                res.append(y)
                break
            elif SequenceMatcher(None, x, y[:10]).ratio() >= 0.6:
                # print(x, "<->", y[:10], SequenceMatcher(None, x, y[:10]).ratio())
                found = True
                res.append(y)
                break
        if not found and len(x) >= 4:
            res.append(x + " @NEWMEMBER")
    return res


def createJson(gpq, names, memberDict):
    for i in range(len(gpq)):
        try:
            score = int(gpq[i])
            if score > 0:
                memberDict[names[i]] = score
        except:
            break


def main():
    memberDict = {}
    members = readMembers("members")
    lst = os.listdir(os.getcwd() + "/scores")
    for i in range(len(lst)):
        if not lst[i].endswith(".png"):
            continue
        files = splitImage(lst[i])
        readNameList = readImg(files[0])
        scores = readImg(files[1])
        actualNames = compNames(readNameList, members)
        createJson(scores, actualNames, memberDict)
    with open(
        "gpq_" + datetime.now().strftime("%m-%d-%Y") + ".json", "w", encoding="utf8"
    ) as f:
        json.dump(memberDict, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    print("Running...")
    main()
    print("Done")
    input("Press enter to close this window...")
