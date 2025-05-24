# Simple AI ALPR (Automatic License Plate Recognition)

Developed by [ARSA Technology](https://arsa.technology), this project is a lightweight, flexible, and efficient license plate recognition system tailored for deployment on a range of devices. It offers multiple inference backends including CPU, CUDA (GPU), and OpenVINO for edge computing.

## 🔍 Features

- ✅ Real-time License Plate Detection
- 🧠 Multiple Inference Backends:
  - `arsalpr_cpu.py`: for CPU-only environments
  - `arsalpr_cuda.py`: for NVIDIA GPU acceleration (CUDA)
  - `arsalpr_vino.py`: for Intel OpenVINO toolkit
- 🔁 Client-server architecture for modular integration
- 🧪 Sandbox mode for custom algorithm testing
- 🚀 Cython-accelerated modules for speed optimization

## 📁 Directory Structure

```
Simple-AI-ALPR-main/
├── arsaLpr/
│   ├── arsalpr_cpu.py
│   ├── arsalpr_cuda.py
│   ├── arsalpr_vino.py
│   ├── client.py
│   ├── server.py
│   ├── sandbox_algorithm.py
│   └── Cython_version/
│       ├── *.py / *.so
│       └── assets/ (model weights and config)
├── setupvars.sh
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/arsa-technology/Simple-AI-ALPR.git
cd Simple-AI-ALPR
```

### 2. Install Dependencies

Make sure Python ≥3.8 is installed.

```bash
pip install -r requirements.txt
```

### 3. Run the System

**Run with CPU:**

```bash
python arsaLpr/arsalpr_cpu.py
```

**Run with CUDA:**

```bash
python arsaLpr/arsalpr_cuda.py
```

**Run with OpenVINO:**

```bash
python arsaLpr/arsalpr_vino.py
```

## 📦 Pre-trained Models

Model configs and weights are stored under:

```
arsaLpr/Cython_version/assets/
```

These include:
- YOLOv3-tiny based configuration for plate detection
- Custom-trained weights for Indonesian license plates
- ESPCN super-resolution model

## 🛠️ Customization

To experiment with custom algorithms, modify:

```bash
arsaLpr/sandbox_algorithm.py
```

To compile Cython modules:

```bash
cd arsaLpr/Cython_version
python setup_compile.py build_ext --inplace
```

## 🧾 License

This project is © 2025 ARSA Technology. All rights reserved. For licensing inquiries, please contact us at [arsa.technology/contact](https://arsa.technology).

## 🌐 About ARSA Technology

ARSA Technology is an Indonesia-based deep tech company developing cutting-edge AI and IoT solutions for smart infrastructure, security, and automation.

Visit us at: [https://arsa.technology](https://arsa.technology)
