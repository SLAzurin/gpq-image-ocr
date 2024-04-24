# from PIL import Image
# from enum import Enum
# import numpy
# from typing import List, Tuple

# import cv2


# class SplitImageType(Enum):
#     LEGACY = 1
#     VIDEO = 2


# def splitImage(
#     im: Image.Image, type: SplitImageType = SplitImageType.LEGACY
# ) -> Tuple[Image.Image, Image.Image]:
#     if type == SplitImageType.LEGACY:
#         resized = im.resize((528, 642))
#         l1 = 45
#         l2 = 364
#         r1 = 120
#         r2 = 420
#         t = 85
#         b = 500
#         im1 = resized.crop((l1, t, r1, b))
#         im2 = resized.crop((l2, t, r2, b))
#         return im1, im2
#     elif type == SplitImageType.VIDEO:
#         resized = im
#         diff = -45
#         l1 = 45 + diff
#         l2 = 364 + diff
#         r1 = 120 + diff
#         r2 = 420 + diff
#         t = 0
#         b = 408
#         im1 = resized.crop((l1, t, r1, b))
#         im2 = resized.crop((l2, t, r2, b))
#         # cv2.imwrite(
#         #     f"frameFromVid1.png", cv2.cvtColor(numpy.array(im1), cv2.COLOR_RGB2BGR)
#         # )
#         # cv2.imwrite(
#         #     f"frameFromVid2.png", cv2.cvtColor(numpy.array(im2), cv2.COLOR_RGB2BGR)
#         # )
#         # exit(1)
#         return im1, im2


# def videoToImages(path: str) -> List[Image.Image]:
#     cap = cv2.VideoCapture(path)
#     i = 0
#     ret, frame_prev = cap.read()
#     images: List[Image.Image] = [
#         Image.fromarray(cv2.cvtColor(frame_prev, cv2.COLOR_BGR2RGB))
#     ]
#     i = 1
#     while cap.isOpened():
#         ret, frame_cur = cap.read()
#         if ret == False:
#             break
#         diff = cv2.absdiff(frame_prev, frame_cur)
#         mean_diff = diff.mean()
#         if mean_diff > 3:
#             images.append(Image.fromarray(cv2.cvtColor(frame_cur, cv2.COLOR_BGR2RGB)))
#             frame_prev = frame_cur
#         i += 1
#     cap.release()
#     cv2.destroyAllWindows()
#     return images


# if __name__ == "__main__":
#     path = "c:\\Users\\Azuri\\Videos\\proper.mp4"
#     images = videoToImages(path)
#     for i, image in enumerate(images):
#         cv2img = cv2.cvtColor(numpy.array(image), cv2.COLOR_RGB2BGR)
#         im1, im2 = splitImage(image, SplitImageType.VIDEO)
#         cv2.imwrite(
#             f"frameFromVid{i}im1.png", cv2.cvtColor(numpy.array(im1), cv2.COLOR_RGB2BGR)
#         )
#         cv2.imwrite(
#             f"frameFromVid{i}im2.png", cv2.cvtColor(numpy.array(im2), cv2.COLOR_RGB2BGR)
#         )
