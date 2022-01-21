import cv2
import numpy as np
from matplotlib import pyplot as plt

bbox = (50, 60, 700, 700)
img = cv2.imread('platnomor1.png', cv2.IMREAD_COLOR)
bgMask = cv2.imread('blackBg.jpg')

def prosesCrop(frame, kosong, bboxCrop):
    widthImg, heightImg, _ = frame.shape
    widthNet, heightNet = (608, 608)
    x1, y1, x2, y2 = bboxCrop
    croppedImg = frame[y1:y2, x1:x2]
    resizedCrop = cv2.resize(croppedImg, (300, 120))
    kosong[200:320, 200:500] = resizedCrop
    return kosong

gambarJadi = prosesCrop(img, bgMask, bbox)




plt.subplot(121),
plt.imshow(img),
plt.title('Source Image'),
plt.axis('off')

plt.subplot(122),
plt.imshow(gambarJadi),
plt.title('Replaced Image'),
plt.axis('off')
 
plt.show()