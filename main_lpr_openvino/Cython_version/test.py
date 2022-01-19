import arsalpr as ai
import cv2


imgPathList = ['../platnomor.jpg', '../platnomor1.png', '../platnomor2.png', '../platnomor3.png']
for imgPath in imgPathList:
    gambarIn = cv2.imread(imgPath)
    ai.tesProses(gambarIn)
