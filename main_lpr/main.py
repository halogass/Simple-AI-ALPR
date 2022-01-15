import arsalpr as aiLpr
import uvicorn
from fastapi import FastAPI, File, UploadFile
from starlette.responses import RedirectResponse
from pydantic import BaseModel
import numpy as np
import cv2


app_desc = """<h2>Try this app by uploading any image to `lpr/image`</h2>
by Hilmy Izzulhaq"""

lpr = FastAPI(title='ARSA LPR API', description=app_desc)

@lpr.get("/", include_in_schema=False)
async def index():
    return RedirectResponse(url="/docs")

@lpr.post("/lpr/image")
async def predict_api(imOut : int, file: UploadFile = File(...)):
    extension = file.filename.split(".")[-1] in ("jpg", "jpeg", "png")
    if not extension:
        return "Image must be jpg or png format!"
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    processedImg, platNomor, latensi = aiLpr.mainProses(img)

    dictPlatNomor = {}
    jumlahNopol = 0
    for x1, y1, x2, y2, noPol, yakin in platNomor:
        jumlahNopol += 1
        dictPlatNomorKey = "License_Plate_" + str(jumlahNopol)
        dictPlatNomor[dictPlatNomorKey] = {
            'xmin': round(x1, 3), 
            'ymin': round(y1, 3), 
            'xmax': round(x2, 3), 
            'ymax': round(y2, 3), 
            'platnomor': noPol, 
            'skor': round(yakin, 2)
        }

    if imOut == 0:
        return dictPlatNomor
    elif imOut == 1:
        return dictPlatNomor


if __name__ == "__main__":
    uvicorn.run(lpr, debug=True)