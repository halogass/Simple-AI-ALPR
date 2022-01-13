import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
from numba import njit
from PIL import Image

weightFile = 'platnomor.weights'
labelsFile = 'platnomor.labels'
configFile = 'platnomor.cfg'
threshDet = 0.35

def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))


def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = image_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    image = darknet.draw_boxes(detections, image_resized, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections


def convert2relative(image, bbox):
    """
    YOLO format use relative coordinates for annotation
    """
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height

def checkInsidePlat(titikPlat, bboxDigit):
    x1, y1, x2, y2 = titikPlat
    x, y, w, h = bboxDigit
    if (x1 <= x and x <= x2):
        if (y1 <= y and y <= y2):
            return True
    else:
        return False

def prosesDigitPlat(digitPlatNom, keyakinanPlat):
    #print('Terdeteksi : ' + str(digitPlatNom))
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
    #print(outputDigit + ' Keyakinan : ' + str(hasilKeyakinan))
    return outputDigit, hasilKeyakinan
        

#platNomor[0] = [0, 0, 0, 0, 'Plat Nomor', 1]

def parseLPR(detections):
    digitsBuffer = detections
    platNomor = []
    adaDigit = False
    for labelPlat, confidencePlat, bboxPlat in detections:
        if labelPlat == 'NOPOL':
            bufTitikPlat = bbox2points(bboxPlat)
            print('platnomor terdeteksi ' + str(bufTitikPlat))
            digitKe = 1
            bufferDigit = []
            for labelDigit, confidenceDigit, bboxDigit in digitsBuffer:
                if (labelDigit != "NOPOL"):
                    if (checkInsidePlat(bufTitikPlat, bboxDigit)):
                        xDigit, yDigit, wDigit, hDigit = bboxDigit
                        bufferDigit.append([labelDigit, xDigit, (float(confidenceDigit)/100)])
                        if digitKe == 1:
                            digitKe = digitKe + 1
                        elif digitKe != 1:
                            digitKe = digitKe + 1
                        if digitKe >= 2:
                            adaDigit = True
                        else:
                            adaDigit = False
            if adaDigit == True:
                hasilDigit = prosesDigitPlat(bufferDigit, (float(confidencePlat)/100))
                platNomor.append([bufTitikPlat + hasilDigit])
    print(platNomor)

def bbox2points(bbox):
    """
    From bounding box yolo format
    to corner points cv2 rectangle
    """
    x, y, w, h = bbox
    xmin = x - (w / 2)
    xmax = x + (w / 2)
    ymin = y - (h / 2)
    ymax = y + (h / 2)
    return xmin, ymin, xmax, ymax

def save_annotations(name, image, detections, class_names):
    """
    Files saved with image_name.txt and relative coordinates
    """
    file_name = os.path.splitext(name)[0] + ".txt"
    with open(file_name, "w") as f:
        for label, confidence, bbox in detections:
            left, top, right, bottom = bbox2points(bbox)
            #label = class_names.index(label)
            f.write("{} {:.4f} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(label, left, top, right, bottom, float(confidence)))


def main():

    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        configFile,
        labelsFile,
        weightFile,
        batch_size=1
    )

    frame = cv2.imread('platnomor.jpg')
    #vid.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    

    index = 0
    while True:
        #return_value, frame = vid.read()
        #if return_value:
        #    image = Image.fromarray(frame)
        #else:
        #    print('Video has ended or failed, try a different video format!')
        #    break
        heightA, widthA, _ = frame.shape
        dim = (widthA, heightA)
        #prev_time = time.time()
        
        image, detections = image_detection(
            frame, network, class_names, class_colors, threshDet
            )
        #if args.save_labels:
        save_annotations('person.jpg', image, detections, class_names)
        #darknet.print_detections(detections, args.ext_output)
        #fps = vid.get(5)
        #fps = int(vid.get(cv2.CAP_PROP_FPS))
        #fps = int(1/(time.time() - prev_time))
        #print("FPS: {}".format(fps))
        parseLPR(detections)
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        
        cv2.imshow('Inference', resized)
        if cv2.waitKey() & 0xFF == ord('q'):
            break
        index += 1


if __name__ == "__main__":
    main()
