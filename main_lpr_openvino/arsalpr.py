import cv2
import numpy as np
import time

MODEL_WEIGHTS = "yolov4-tiny.weights"
MODEL_CONFIG = "yolov4-tiny.cfg"
CLASS_LABELS = "classes.labels"

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
COLOR = (0, 255, 255)
NETWORK_DIMENSION = (416, 416)

class_names = open(CLASS_LABELS).read().splitlines()

vc = cv2.VideoCapture("/home/arsa/Videos/traffic.mp4")

net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

def detectionEngine(imageIn):
    classes, scores, boxes = model.detect(imageIn, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    detections = []
    for (classid, score, box) in zip(classes, scores, boxes):
        label = class_names[int(classid)]
        detections.append([label, box, score])
    return detections

def checkInsidePlat(titikPlat, bboxDigit):
    x1, y1, x2, y2 = titikPlat
    x, y, w, h = bboxDigit
    if (x1 <= x and x <= x2):
        if (y1 <= y and y <= y2):
            return True
    else:
        return False

def prosesSortDigitPlat(digitPlatNom):
    sortedList = sorted(digitPlatNom, key=lambda x: x[1], reverse=False)
    bufferDigit = []
    totalKeyakinan = 0
    loopKe = 0
    for digit, titik, keyakinan in sortedList:
        bufferDigit.append(digit)
        totalKeyakinan = totalKeyakinan + keyakinan
        loopKe = loopKe + 1
    totalKeyakinan = totalKeyakinan / loopKe
    hasilKeyakinan = totalKeyakinan
    outputDigit = "".join(bufferDigit)
    return outputDigit, hasilKeyakinan

def prosesCrop(frame, bboxCrop):
    x1, y1, x2, y2 = bboxCrop
    rx1, ry1, rx2, ry2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    imgCroped = frame[ry1:ry2, rx1:rx2]

    imgBg = np.zeros((1000,1000,3), dtype=np.uint8)
    resizedDimPlat = (1000, 300)

    resizedImPlat = cv2.resize(imgCroped, resizedDimPlat)
    imgBg[333:633, 0:1000] = resizedImPlat
    
    return imgBg

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    start = time.time()
    detections = detectionEngine(frame)
    end = time.time()

    start_drawing = time.time()
    for classlabel, box, score in detections:
        color = COLOR
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, "{}".format(classlabel), (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()
    
    fps_label = "FPS: %.2f (excluding drawing time of %.2fms)" % (1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.imshow("detections", frame)
