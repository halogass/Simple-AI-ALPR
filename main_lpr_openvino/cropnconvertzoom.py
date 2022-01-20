import cv2 
import numpy as np 
import matplotlib.pyplot as plt
import os

inputDir = '/home/arsa/Pictures/Datasets/OCRLpr/Original'
outputDir = '/home/arsa/Pictures/Datasets/OCRLpr/Cropped'

inputDirList = os.listdir(inputDir)

def cariImagePath(fileWithNoExt):
    if os.path.exists(fileWithNoExt + '.png'):
        imagepath = (fileWithNoExt + '.png')
        #print('File PNG')
        return imagepath
    elif os.path.exists(fileWithNoExt + '.jpg'):
        imagepath = (fileWithNoExt + '.jpg')
        #print('File JPG')
        return imagepath
    elif os.path.exists(fileWithNoExt + '.jpeg'):
        imagepath = (fileWithNoExt + '.jpeg')
        #print('File JPEG')
        return imagepath


def loadProcessConvertBbox(labelInPath, dimFrame):
    labelIn = np.genfromtxt(labelInPath, dtype=None)
    print(labelIn)
    height, width, _ = dimFrame
    centeredBbox = []
    for className, xCenter, yCenter, wBox, hBox in labelIn:
        bufferArrayCentered = [0 for i in range(5)]
        bufferArrayCentered[0] = className
        bufferArrayCentered[1] = xCenter * width
        bufferArrayCentered[2] = yCenter * height
        bufferArrayCentered[3] = wBox * width
        bufferArrayCentered[4] = hBox * height
        centeredBbox.append(bufferArrayCentered)
    centeredBbox_np = np.array(centeredBbox)
    print('Converted to real : ')
    print(centeredBbox_np)
    return centeredBbox_np


def checkInsidePlat(titikPlat, bboxDigit):
    x1, y1, x2, y2 = titikPlat
    x, y, w, h = bboxDigit
    if (x1 <= x and x <= x2):
        if (y1 <= y and y <= y2):
            return True
    else:
        return False

def rubahRelatifKoor(titik, dimAwal, dimAkhir):
    x1A, y1A, x2A, y2A = titik
    widthAwal, heightAwal = dimAwal
    widthAkhir, heightAkhir = dimAkhir
    x1B = (x1A / widthAwal) * widthAkhir
    x2B = (x2A / widthAwal) * widthAkhir
    y1B = (y1A / heightAwal) * heightAkhir
    y2B = (y2A / heightAwal) * heightAkhir
    return x1B, y1B, x2B, y2B

def bbox2points(bbox):
    x, y, w, h = bbox
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax

def points2YoloList(pointsList, irengImPlat):
    hPlat, wPlat, _ = irengImPlat.shape
    outCenList = []
    for label, x1, y1, x2, y2 in pointsList:
        xCen = ((x1 + ((x2 - x1) / 2)) / wPlat)
        yCen = ((y1 + ((y2 - y1) / 2)) / hPlat)
        wCen = ((x2 - x1) / wPlat)
        hCen = ((y2 - y1) / hPlat)
        outCenList.append([label, xCen, yCen, wCen, hCen])
    listCenFinal = np.array(outCenList)
    return listCenFinal

def showHasil(imgHasil):
    cv2.imshow('Inference', imgHasil)
    cv2.waitKey() & 0xFF == ord('q')

def prosesCrop(frame, bboxCrop):
    #print(width, height)
    #print(bboxCrop)
    x1, y1, x2, y2 = bboxCrop
    rx1, ry1, rx2, ry2 = int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))
    #print(rx1, ry1, rx2, ry2)
    imgCroped = frame[ry1:ry2, rx1:rx2]
    return imgCroped

def drawPlat(dataPlatNom, gambar, class_names):
    warnaPlat = (255, 255, 0)
    yakin = 0.1
    for listPlat in dataPlatNom:
        nomor, x1, y1, x2, y2 = listPlat
        cv2.rectangle(gambar, (int(round(x1)), int(round(y1))), (int(round(x2)), int(round(y2))), warnaPlat, 4)
        cv2.rectangle(gambar, (int(round(x1)), int(round(y2))), (int(round(x1 + 80)), int(round(y2 + 50))), warnaPlat, -1)
        cv2.putText(gambar, "{}".format(class_names[int(nomor)]),
                    (int(round(x1 + 5)), int(round(y2)) + 40), cv2.FONT_HERSHEY_SIMPLEX, 1.6,
                    (0, 0, 0), 4)
    return gambar

def tambahiIreng(resizedImg, resizedLabel):
    hImIn, wImIn, _ = resizedImg.shape
    imgBg = np.zeros((1000,1000,3), dtype=np.uint8)
    imgBg[333:633, 0:1000] = resizedImg
    buffJumlahY = []
    for label, x1, y1, x2, y2 in resizedLabel:
        y1 = y1 + 333
        y2 = y2 + 333
        buffJumlahY.append([label, x1, y1, x2, y2])
    outLabel = np.array(buffJumlahY)
    return imgBg, outLabel

def processCropBboxPlat(imageIn, labelCenIn, class_names):
    buffImg = imageIn
    print('label yg masuk : ')
    print(labelCenIn)
    finalImgLabelList = []
    terdetekPlat = False
    for className, x, y, w, h in labelCenIn:
        bboxCenPlat = x, y, w, h
        bboxPlat = bbox2points(bboxCenPlat)
        bufferDigit = []
        bufferPlatDig = []
        if className == 24:
            print('Terdeteksi platnomor : ')
            cropedImgPl = prosesCrop(imageIn, bboxPlat)
            hCropedPlat, wCroppedPlat, _ = cropedImgPl.shape
            dimCroppedPlat = wCroppedPlat, hCropedPlat
            print(bboxPlat)
            #plt.imshow(cropedImgPl)
            #plt.show()
            jumlahDigit = 0
            #x1Plat, y1Plat, x2Plat, y2Plat = bboxPlat
            #bufferDigit.append([24, 0, 0, w, h])
            for digitName, xCen, yCen, wBox, hBox in labelCenIn:            
                bboxCoorDigit = xCen, yCen, wBox, hBox
                if not digitName == 24:
                    if checkInsidePlat(bboxPlat, bboxCoorDigit):
                        jumlahDigit += 1
                        xCenOut = xCen - bboxPlat[0]
                        yCenOut = yCen - bboxPlat[1]
                        bufferDigit.append([digitName, xCenOut, yCenOut, wBox, hBox])
                        #print('Terdeteksi digit platnomor')
            if jumlahDigit >= 2:
                terdetekPlat = True
                bufferDigit_np = np.array(bufferDigit)
                print('Hasil pemotongan bbox :')
                print(bufferDigit_np)
                resizedDimPlat = (1000, 300)
                coorRealList = []
                heightPlatCroped, widthPlatCroped, _ = cropedImgPl.shape
                titikPlat = (0, 0, widthPlatCroped, heightPlatCroped)
                x10P, y10P, x20P, y20P = rubahRelatifKoor(titikPlat, dimCroppedPlat, resizedDimPlat)
                coorRealList.append([24, 0, 0, x20P, y20P])
                for labelNum, x0, y0, w0, h0 in bufferDigit_np:
                    bboxIn0 = x0, y0, w0, h0
                    titik = bbox2points(bboxIn0)
                    x10, y10, x20, y20 = rubahRelatifKoor(titik, dimCroppedPlat, resizedDimPlat)
                    coorRealList.append([labelNum, x10, y10, x20, y20])
                coorRealList_np = np.array(coorRealList)
                resizedImPlat = cv2.resize(cropedImgPl, resizedDimPlat)
                irengImPlat, irengLabel = tambahiIreng(resizedImPlat, coorRealList_np)
                print('Final BBOX Akhir : ')
                print(irengLabel)
                labelYoloFormat = points2YoloList(irengLabel, irengImPlat)
                finalImgLabelList.append([irengImPlat, labelYoloFormat, irengLabel])
                #drawedImgPlat = drawPlat(irengLabel, irengImPlat, class_names)
                #plt.imshow(drawedImgPlat)
                #plt.show()
    return finalImgLabelList
                
def saveImLabelFinal(filenameWPath, imgFinal, labelYolo):
    cv2.imwrite((filenameWPath + '_cropped_' + '.jpg'), imgFinal)
    with open((filenameWPath + '_cropped_' + '.txt'), 'w') as saveLabelTxt:
        for label, x, y, w, h in labelYolo:
            saveLabelTxt.write("{} {:.20f} {:.20f} {:.20f} {:.20f}\n".format(int(label), x, y, w, h))



def main():
    print('Mari kita mulai')
    jumlahFile = 0
    for fileTxtIn in inputDirList:
        if fileTxtIn.endswith(".txt"):
            jumlahFile += 1
            noExtension = os.path.splitext((inputDir + '/' + fileTxtIn))[0]
            noExtOut = os.path.splitext((outputDir + '/' + fileTxtIn))[0]
            print(noExtension)
            labelPath = (noExtension + '.txt')
            imagePath = cariImagePath(noExtension)
            imageIn = cv2.imread(imagePath)
            #image_in = cv2.cvtColor(imageIn, cv2.COLOR_BGR2RGB)
            image_in = imageIn
            class_names = open('platnomor.labels').read().splitlines()
            heightIm, widthIm, _ = image_in.shape
            dimIm = widthIm, heightIm
            print('Dimensi Gambar : ' + str(dimIm))
            labelCenIn = loadProcessConvertBbox(labelPath, image_in.shape)
            processedFinal = processCropBboxPlat(image_in, labelCenIn, class_names)
            if processedFinal:
                for imageFinal, labelFinalYolo, labelFinalCv in processedFinal:
                    print('Hasil Final : ')
                    print(labelFinalYolo)
                    saveImLabelFinal(noExtOut, imageFinal, labelFinalYolo)
                    #drawedImgPlat = drawPlat(labelFinalCv, imageFinal, class_names)
                    #plt.imshow(drawedImgPlat)
                    #plt.show()
            #print(dimIm)
            #print(labelIn)
            #plt.imshow(imageIn)
            #plt.show()
        #break

    #print(jumlahFile)

main()