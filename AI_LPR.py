import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
from PIL import Image

weightFile = 'platnomor.weights'
labelsFile = 'platnomor.labels'
configFile = 'platnomor.cfg'
threshDet = 0.35


def image_detection(image_path, network, class_names, class_colors, thresh):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    networkDimension = [width, height]
    darknet_image = darknet.make_image(width, height, 3)

    image = image_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    #image = darknet.draw_boxes(detections, image_resized, class_colors)
    image = image_resized
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections, networkDimension

def checkInsidePlat(titikPlat, bboxDigit):
    x1, y1, x2, y2 = titikPlat
    x, y, w, h = bboxDigit
    if (x1 <= x and x <= x2):
        if (y1 <= y and y <= y2):
            return True
    else:
        return False

def prosesDigitPlat(digitPlatNom, keyakinanPlat):
    sortedList = sorted(digitPlatNom, key=lambda x: x[1], reverse=False)
    bufferDigit = []
    totalKeyakinan = 0
    loopKe = 0
    for digit, titik, keyakinan in sortedList:
        bufferDigit.append(digit)
        totalKeyakinan = totalKeyakinan + keyakinan
        loopKe = loopKe + 1
    totalKeyakinan = totalKeyakinan / loopKe
    hasilKeyakinan = (keyakinanPlat + totalKeyakinan) / 2
    outputDigit = "".join(bufferDigit)
    return outputDigit, hasilKeyakinan


def parseLPR(detections, dimDarknet, dimAsli):
    digitsBuffer = detections
    platNomor = []
    for labelPlat, confidencePlat, bboxPlat in detections:
        if labelPlat == 'NOPOL':
            bufTitikPlat = bbox2points(bboxPlat)
            digitKe = 1
            bufferDigit = []
            for labelDigit, confidenceDigit, bboxDigit in digitsBuffer:
                if (labelDigit != "NOPOL"):
                    if (checkInsidePlat(bufTitikPlat, bboxDigit)):
                        xDigit, yDigit, wDigit, hDigit = bboxDigit
                        bufferDigit.append([labelDigit, xDigit, (float(confidenceDigit)/100)])
                        digitKe = digitKe + 1
            if digitKe >= 2:
                hasilDigit = prosesDigitPlat(bufferDigit, (float(confidencePlat)/100))
                koorPlatAseli = rubahRelatifKoor(bufTitikPlat, dimDarknet, dimAsli)
                arrayPlat = koorPlatAseli + hasilDigit
                platNomor.append(arrayPlat)
    return platNomor

def rubahRelatifKoor(titik, dimDarknet, dimAseli):
    x1A, y1A, x2A, y2A = titik
    widthDarknet, heightDarknet = dimDarknet
    widthAseli, heightAseli = dimAseli
    x1B = (x1A / widthDarknet) * widthAseli
    x2B = (x2A / widthDarknet) * widthAseli
    y1B = (y1A / heightDarknet) * heightAseli
    y2B = (y2A / heightDarknet) * heightAseli
    return x1B, y1B, x2B, y2B


def drawPlat(dataPlatNom, gambar):
    for listPlat in dataPlatNom:
        x1, y1, x2, y2, nomor, yakin = listPlat
        cv2.rectangle(gambar, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), (255, 0, 0), 1)
        cv2.rectangle(gambar, (int(round(x1)), int(round(y1 - 20))), (int(round(x1 + 120)), int(round(y1))), (255, 0, 0), -1)
        cv2.putText(gambar, "{} [{:.2f}]".format(nomor, float(yakin)),
                    (int(round(x1 + 5)), int(round(y1)) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    return gambar

def bbox2points(bbox):
    x, y, w, h = bbox
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax

def main():
    random.seed(3)
    network, class_names, class_colors = darknet.load_network(
        configFile,
        labelsFile,
        weightFile,
        batch_size=1
    )

    frame = cv2.imread('platnomor.jpg')

    while True:
        heightA, widthA, _ = frame.shape
        dim = (widthA, heightA)
        prev_time = time.time()
        
        image, detections, networkDimension = image_detection(
            frame, network, class_names, class_colors, threshDet
            )

        nomorPlat = parseLPR(detections, networkDimension, dim)
        gambarA = drawPlat(nomorPlat, frame)
        latencyAi = (time.time() - prev_time)
        print('Latensi : ' + str(latencyAi))
        cv2.imshow('Inference', gambarA)
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        break


if __name__ == "__main__":
    main()
