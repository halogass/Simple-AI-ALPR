# Simple AI ALPR (Automatic License Plate Recognition)

This is a simple AI-based License Plate Recognition (LPR) project developed by [ARSA Technology](https://arsa.technology). The project uses deep learning models for license plate detection and recognition, and includes optional super-resolution for improved accuracy on low-resolution images.

## Features

- License plate detection and character recognition using YOLO-based models.
- Optional super-resolution using ESPCN for enhanced image quality.
- REST API server for easy integration.
- Example client for testing with images.
- Supports PNG and JPEG image formats.

## Project Structure

```
arsaLpr/
  arsalpr_cpu.py
  arsalpr_cuda.py
  arsalpr_vino.py
  client.py
  sandbox_algorithm.py
  server.py
  assets/
    ESPCN_x4.pb
    platnomor-tiny.cfg
    platnomor-tiny.weights
    platnomor-train_best.weights
    platnomor.labels
    response.jpg
img_asset/
  platnomor.jpg
  platnomor1.png
  platnomor3.png
setupvars.sh
```

## Getting Started

### Requirements

- Python 3.7+
- OpenCV (with dnn and dnn_superres modules)
- FastAPI
- Uvicorn
- NumPy
- Requests
- Matplotlib

Install dependencies:

```sh
pip install opencv-python-headless fastapi uvicorn numpy requests matplotlib
```

### Running the API Server

Start the API server:

```sh
cd arsaLpr
python server.py
```

The server will be available at `http://127.0.0.1:5402`. API documentation is available at `/docs`.

### Using the Client

Test the API with example images:

```sh
cd arsaLpr
python client.py
```

### API Usage

Send a POST request to `/v0/lpr` with an image file:

- `imOut`: `true` to include the result image (base64), `false` for JSON only.
- `superRes`: `true` to enable super-resolution, `false` for faster processing.

Example using `curl`:

```sh
curl -F "file=@../img_asset/platnomor.jpg" "http://127.0.0.1:5402/v0/lpr?imOut=true&superRes=false"
```

## Model Files

- `platnomor-tiny.cfg`, `platnomor-tiny.weights`, `platnomor-train_best.weights`: YOLO model files for detection.
- `platnomor.labels`: Class labels for detection.
- `ESPCN_x4.pb`: Super-resolution model.

## License

This project is licensed under the Apache 2.0 License.

---

Developed by [ARSA Technology](https://arsa.technology)
