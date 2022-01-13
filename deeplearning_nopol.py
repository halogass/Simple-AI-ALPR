import os
import glob
import random
import darknet
import time
import cv2
import sys
import numpy as np

def image_detection(image_path, network, class_names, class_colors, thresh):
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    darknet.free_image(darknet_image)
    #image = darknet.draw_boxes(detections, image_resized, class_colors)
    image = image_resized
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), detections

def convert2relative(image, bbox):
    x, y, w, h = bbox
    height, width, _ = image.shape
    return x/width, y/height, w/width, h/height


def save_ocr_result(name, image, detections, class_names, fileOut2):
    fileOut3 = os.path.splitext((fileOut2))[0]
    file_name = os.path.splitext(name)[0] + ".txt"
    with open("buffer0ocr.txt", "w") as f:
        for label, confidence, bbox in detections:
            x, y, w, h = convert2relative(image, bbox)
            label = class_names.index(label)
            labeltext = class_names[int(label)]
            if(labeltext != "NOPOL"):
                f.write("{:.4f} {}\n".format(x, labeltext))
            
            
    with open("buffer0ocr.txt",'r') as first_file:
        rows = first_file.readlines()
        sorted_rows = sorted(rows, key=lambda x: float(x.split()[0]), reverse=False)
        with open("buffer1ocr.txt",'w') as second_file:
            for row in sorted_rows:
                rowa = row.split(" ", 1)[1]
                second_file.write(rowa)
    os.remove("buffer0ocr.txt")
    with open("buffer1ocr.txt") as f1a, open((fileOut3 + ".txt"), 'w') as f2a:
        f2a.write("".join(line.rstrip('\n') for line in f1a))
        print("".join(line.rstrip('\n') for line in f1a))
    os.remove("buffer1ocr.txt")
    with open((fileOut3 + ".txt"), 'r') as printhasil:
        printhasila = printhasil.readlines()
        print(printhasila)
    os.remove(fileOut2)
    

def resource_path(relative_path):
    base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base_path, relative_path)


def main():
    random.seed(3)
    network, class_names, class_colors = darknet.load_network(
        resource_path("platnomor.cfg"),
        resource_path("platnomor.data"),
        resource_path("platnomor.weights"),
        batch_size=1
    )
    print("Neural Network Loaded")
    index = 0
    while True:
        loopSearchFile = 1
        input_image_file = os.listdir('/var/www/AI/OCR/Nopol')
        print(input_image_file)
        for fileIn in input_image_file:
            if 'jpg' in fileIn or 'png' in fileIn or 'jpeg' in fileIn:
                time.sleep(0.5)
                print(fileIn)
                image_name = ('/var/www/AI/OCR/Nopol/' + fileIn)
                fileOut = os.path.splitext(('/var/www/AI/OCR/Nopol/' + fileIn))[0]
                image, detections = image_detection(
                    image_name, network, class_names, class_colors, 0.3
                    )
                print("Detection Process Done")
                save_ocr_result(image_name, image, detections, class_names, image_name)
                darknet.print_detections(detections, True)
                print('\n\n\n')
                index += 1
        #break


if __name__ == "__main__":
    main()
