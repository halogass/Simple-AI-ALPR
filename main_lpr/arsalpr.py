import random
import arsai
import time
import cv2

imgPathList = ['platnomor.jpg', 'platnomor1.png', 'platnomor2.png', 'platnomor3.png']
weightFile = 'platnomor.weights'
labelsFile = 'platnomor.labels'
configFile = 'platnomor.cfg'
threshDet = 0.3

platColor = (255, 0, 100)

modeNet = 1

network, class_names = arsai.load_network(
    configFile,
    labelsFile,
    weightFile,
    batch_size=1
    )

def image_detection(image_path, network, class_names, thresh):
    width = arsai.network_width(network)
    height = arsai.network_height(network)
    networkDimension = [width, height]
    arsai_image = arsai.make_image(width, height, 3)

    image = image_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    arsai.copy_image_from_bytes(arsai_image, image_resized.tobytes())
    detections = arsai.detect_image(network, class_names, arsai_image, thresh=thresh)
    arsai.free_image(arsai_image)
    return detections, networkDimension

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
    hasilKeyakinan = totalKeyakinan
    outputDigit = "".join(bufferDigit)
    return outputDigit, hasilKeyakinan


def parseLPR(detections, dimArsai, dimAsli):
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
                koorPlatAseli = rubahRelatifKoor(bufTitikPlat, dimArsai, dimAsli)
                arrayPlat = koorPlatAseli + hasilDigit
                platNomor.append(arrayPlat)
    return platNomor

def rubahRelatifKoor(titik, dimArsai, dimAseli):
    x1A, y1A, x2A, y2A = titik
    widthArsai, heightArsai = dimArsai
    widthAseli, heightAseli = dimAseli
    x1B = (x1A / widthArsai) * widthAseli
    x2B = (x2A / widthArsai) * widthAseli
    y1B = (y1A / heightArsai) * heightAseli
    y2B = (y2A / heightArsai) * heightAseli
    return x1B, y1B, x2B, y2B


def drawPlat(dataPlatNom, gambar, warnaPlat):
    for listPlat in dataPlatNom:
        x1, y1, x2, y2, nomor, yakin = listPlat
        cv2.rectangle(gambar, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), warnaPlat, 1)
        cv2.rectangle(gambar, (int(round(x1)), int(round(y1 - 20))), (int(round(x1 + 120)), int(round(y1))), warnaPlat, -1)
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

def prosesGambar(frame, network, class_names, class_colors, threshDet):
    prev_time = time.time()
    heightFrame, widthFrame, _ = frame.shape
    dimFrame = (widthFrame, heightFrame)
    detections, networkDimension = image_detection(frame, network, class_names, threshDet)
    nomorPlat = parseLPR(detections, networkDimension, dimFrame)
    gambarA = drawPlat(nomorPlat, frame, platColor)
    latencyAi = (time.time() - prev_time) * 1000
    return gambarA, nomorPlat, latencyAi

def showHasil(imgHasil):
    cv2.imshow('Inference', imgHasil)
    cv2.waitKey() & 0xFF == ord('q')

def tesProses(inputIm):
    random.seed(3)
    processedImg, platNomor, latensi = prosesGambar(inputIm, network, class_names, platColor, threshDet)

    print('Latensi (ms) : ' + str(latensi))
    platnomdetek = []
    for items in platNomor:
        platnomdetek.append(items[4])
    print('Platnomor : ' + str(platnomdetek))
    #showHasil(processedImg)


if __name__ == "__main__":
    i = 0
    for imgPath in imgPathList:
        gambarIn = cv2.imread(imgPath)
        tesProses(gambarIn)
