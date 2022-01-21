import cv2
import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt

MODEL_WEIGHTS = 'assets/platnomor-train_best.weights'
MODEL_CONFIG = "assets/platnomor-tiny.cfg"
CLASS_LABELS = "assets/platnomor.labels"

SUPERRES_MODEL = "assets/ESPCN_x4.pb"

CONFIDENCE_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4
COLOR = (255, 0, 0)
NETWORK_DIMENSION = (416, 416)

# load super resolution model then set the backend and target
superres = cv2.dnn_superres.DnnSuperResImpl_create()
superres.readModel(SUPERRES_MODEL)
superres.setModel("espcn",4)
superres.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
superres.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# load object detection model then set the backend and target
net = cv2.dnn.readNetFromDarknet(MODEL_CONFIG, MODEL_WEIGHTS)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_INFERENCE_ENGINE)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# Initiate detection model
model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

# Load classes
class_names = open(CLASS_LABELS).read().splitlines()

# Detection function
def detectionEngine(imageIn):
    classes, scores, boxes = model.detect(imageIn, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    detections = []
    for (classid, score, box) in zip(classes, scores, boxes):
        label = class_names[int(classid)]
        detections.append([label, box, score])
    return detections

# Sort digits inside a plate using the x axis center point of their bbox
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
    hasilKeyakinan = round(float(totalKeyakinan), 2)
    outputDigit = "".join(bufferDigit)
    return outputDigit, hasilKeyakinan

def prosesCrop(bboxCrop, imageCropIn):
    x1, y1, x2, y2 = bboxCrop
    rx1, ry1, rx2, ry2 = int(x1), int(y1), int(x2), int(y2)
    imgCroped = imageCropIn[ry1:ry2, rx1:rx2].copy()
    imgBg = np.zeros((1000,1000,3), dtype=np.uint8)
    resizedDimPlat = (1000, 300)
    resizedImPlat = cv2.resize(imgCroped, resizedDimPlat)
    imgBg[333:633, 0:1000] = resizedImPlat
    return imgBg

def drawPlat(dataPlatNom, gambar, warnaPlat):
    for bbox, nomor, yakin in dataPlatNom:
        x1, y1, x2, y2 = bbox
        cv2.rectangle(gambar, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), warnaPlat, 1)
        cv2.rectangle(gambar, (int(round(x1)), int(round(y1 - 20))), (int(round(x1 + 120)), int(round(y1))), warnaPlat, -1)
        cv2.putText(gambar, "{} [{:.2f}]".format(nomor, float(yakin)),
                    (int(round(x1 + 5)), int(round(y1)) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 255), 1)
    return gambar

def showHasil(imgHasil):
    show_rgb = cv2.cvtColor(imgHasil, cv2.COLOR_BGR2RGB)
    plt.imshow(show_rgb)
    plt.show()
    #cv2.imshow('Inference', imgHasil)
    #cv2.waitKey() & 0xFF == ord('q')

def superresAlgorith(image_In):
    result_img = superres.upsample(image_In)
    return result_img

def engineLprAi(buffImage_In, superres_mode):
    prev_time = time.time()
    if superres_mode == True:
        image_In = superresAlgorith(buffImage_In)
    elif superres_mode == False:
        image_In = buffImage_In
    lprPlatnomor = []
    detections = detectionEngine(image_In)
    for labelPlat, bboxRawPlat, confidencePlat in detections:
        if labelPlat == "NOPOL":
            x1P, y1P, wP, hP = bboxRawPlat
            bboxPlat = x1P, y1P, (x1P + wP), (y1P + hP)
            #print('Terdeteksi platnomor, posisi :')
            #print(bboxPlat)
            platCroped = prosesCrop(bboxPlat, image_In)
            detectionsDigit = detectionEngine(platCroped)
            digitKe = 0
            bufferDigit = []
            for labelDigit, bboxDigit, confidenceDigit in detectionsDigit:
                if not labelDigit == "NOPOL":
                    x1D, y1D, wD, hD = bboxDigit
                    xCen = (x1D + (wD / 2))
                    bufferDigit.append([labelDigit, xCen, confidenceDigit])
                    digitKe += 1
            if digitKe >= 2:
                digitTerurut, keyakinanDigit = prosesSortDigitPlat(bufferDigit)
                lprPlatnomor.append([bboxPlat, digitTerurut, keyakinanDigit])
    latencyAi = (time.time() - prev_time) * 1000
    drawed_img = drawPlat(lprPlatnomor, image_In, COLOR)
    return lprPlatnomor, latencyAi, drawed_img

def mainProses(ImgPlatIn, supresMode):
    platnomorHasil, latensi, drawed_img = engineLprAi(ImgPlatIn, supresMode)
    return drawed_img, platnomorHasil, latensi

def testPict():
    imgIn = cv2.imread('../img_asset/platnomor3.png')

    platnomorHasil, latensi, drawed_img = engineLprAi(imgIn, False)
    print(platnomorHasil)
    print('Latensi : ' + str(latensi))
    showHasil(drawed_img)

def testVid():
    videoIn = cv2.VideoCapture("/home/arsa/Videos/m3_edited.mp4")
    while cv2.waitKey(1) < 1:
        (grabbed, frameIn) = videoIn.read()
        if not grabbed:
            exit()
        start = time.time()
        platnomorHasil, latensi, drawImg = engineLprAi(frameIn, False)
        end = time.time()
        fps_label = "FPS: %.2f" % (1 / (end - start))
        cv2.putText(drawImg, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("detections", drawImg)


#testPict()
