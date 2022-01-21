from xmlrpc.client import Boolean
import arsalpr_cuda as aiLpr
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
import numpy as np
import cv2
import base64

hostIpAddr = "127.0.0.1"

app_desc = """<h2>Try this app by uploading any image to `lpr/image`</h2>
Set query imOut= true or false
false mean no result image, true mean with result image<br>
result image is an output image with license plate bounding box and label already drawed<br>
result_img is encoded in base64 (string)<br>
<br>
you can also you super resolution algorith to detect far objects, but with slower processing time<br>
by Hilmy Izzulhaq"""

lpr = FastAPI(
    title="ARSA LPR API DEMO",
    description=app_desc,
    version="0.0.1",
    contact={
        "name": "Hilmy Izzulhaq",
        "email": "halogas.ijul@gmail.com",
    },
    license_info={
        "name": "Apache 2.0",
        "url": "https://www.apache.org/licenses/LICENSE-2.0.html",
    },
)

@lpr.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@lpr.post("/v0/lpr", tags=["LPR"])
async def lpr_api(imOut : Boolean, superRes : Boolean,  file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processedImg, platNomor, latensi = aiLpr.mainProses(img, superRes)

    tipeOcr = 'license_plate_recognition'

    print(platNomor)
    dictPlatNomor = {}
    jumlahNopol = 0
    statusBaca = 'failed'
    for bbox, noPol, yakin in platNomor:
        x1, y1, x2, y2 = bbox
        jumlahNopol += 1
        statusBaca = 'success'
        dictPlatNomorKey = str(jumlahNopol)
        dictPlatNomor[dictPlatNomorKey] = {
            'xmin':round(float(x1)), 
            'ymin':round(float(y1)), 
            'xmax':round(float(x2)), 
            'ymax':round(float(y2)), 
            'license_plate':str(noPol), 
            'skor':round(float(yakin), 2)
        }

    dictOutput = {
        'status':statusBaca,
        'ai_type':tipeOcr,
        'processing_time(ms)':round(latensi, 3),
        'result':dictPlatNomor,
        'filename':file.filename
    }

    if imOut == False:
        return dictOutput
    elif imOut == True:
        _, encoded_img = cv2.imencode('.PNG', processedImg)
        encoded_imgOut = base64.b64encode(encoded_img)
        dictOutput['result_img'] = encoded_imgOut
        return dictOutput


if __name__ == "__main__":
    uvicorn.run(lpr, debug=True, host=hostIpAddr, port=5402, headers=[("server", "arsa-ai_server-1"), ("AI-Developer", "ARSA-Technology")])