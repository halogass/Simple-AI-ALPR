from pprint import pprint
import requests
import base64
import numpy as np
import cv2

intloopRequest = 0

def showHasil(imgHasil):
    cv2.imshow('Inference', imgHasil)
    cv2.waitKey() & 0xFF == ord('q')

while intloopRequest <= 3:
    intloopRequest += 1
    files = {'file': open('platnomor.jpg', 'rb')}
    response = requests.post('http://api.arsa.technology:5402/v0/lpr?imOut=1', files=files)
    inputRes = response.json()
    inputImgRaw = inputRes['result_img']
    processTime = inputRes['processing_time(ms)']
    nparr = np.frombuffer(base64.b64decode(inputImgRaw), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print(processTime)
    #showHasil(img)


