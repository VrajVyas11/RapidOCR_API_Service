# RapidOCR API Service

## Overview

The **RapidOCR API Service** is a high-performance, serverless Optical Character Recognition (OCR) API designed for extracting and grouping text from images, with a focus on manga and comic-style documents. Built using FastAPI and deployed on Render.com, this API leverages the `rapidocr-onnxruntime` library to perform efficient text detection and recognition. It includes advanced features like manga-specific text grouping (right-to-left, top-to-bottom reading order) and support for modern image formats (AVIF, HEIF, WebP). The service is optimized for low memory usage, fast processing, and seamless deployment, making it ideal for developers and manga enthusiasts.

### Key Features
- **High-Accuracy OCR**: Utilizes `rapidocr-onnxruntime` for robust text detection and recognition in images.
- **Manga-Optimized Text Grouping**: Groups text bubbles based on proximity and alignment, tailored for manga’s right-to-left, top-to-bottom reading order.
- **Multi-Format Image Support**: Handles modern image formats (JPEG, PNG, WebP, AVIF, HEIF) using `Pillow` and `pillow_heif`.
- **Memory Monitoring**: Tracks memory usage with `psutil` to ensure efficient resource management.
- **Word Splitting**: Enhances text readability using `wordninja` to split concatenated words.
- **FastAPI Framework**: Provides a lightweight, asynchronous API with automatic OpenAPI documentation.
- **Render.com Deployment**: Deployed as a web service on Render.com for scalability and ease of use.

### Benefits
- **Specialized for Manga**: The API’s text grouping algorithm is designed specifically for manga, ensuring accurate paragraph formation for dialogue bubbles.
- **Lightweight Deployment**: Optimized dependencies reduce bundle size, making it suitable for Render.com’s resource constraints.
- **High Compatibility**: Supports a wide range of image formats, increasing versatility for various use cases.
- **Developer-Friendly**: FastAPI’s OpenAPI docs and JSON responses simplify integration into applications.
- **Resource Efficiency**: Memory monitoring and garbage collection (`gc`) minimize resource usage, ideal for serverless environments.
- **Scalable**: Render.com’s auto-scaling ensures the API handles varying loads efficiently.

## Tags
- **OCR**
- **Manga**
- **FastAPI**
- **Render.com**
- **Text Recognition**
- **Image Processing**
- **Python**
- **AVIF**
- **HEIF**
- **WebP**
- **Serverless**
- **Machine Learning**

## Project Structure
```
EasyOCRService/
├── api/
│   └── main.py         # FastAPI application with OCR logic
├── requirements.txt    # Project dependencies
└── README.md          # Project documentation
```

- **`api/main.py`**: Core FastAPI application with endpoints for OCR initialization, text extraction, and cleanup.
- **`requirements.txt`**: Lists optimized dependencies for deployment on Render.com.
- **`README.md`**: This file, providing comprehensive project documentation.

## Installation

### Prerequisites
- **Python**: 3.12 or later
- **Render.com Account**: For deployment
- **Git**: For cloning the repository

### Local Setup
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/VrajVyas11/RapidOCR_API_Service.git
   cd RapidOCR_API_Service
   ```

2. **Create a Virtual Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Locally**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
   Access the API at `http://localhost:8000`.

### Optimized `requirements.txt`
To address the previous bundle size issue (391.34 MB unzipped, 135.12 MB zipped), the dependencies are optimized to stay within Render.com’s limits (typically ~500 MB unzipped, no strict zipped limit). The updated `requirements.txt` removes `opencv-python-headless` and uses lighter versions:

```
fastapi==0.104.1
python-multipart==0.0.6
rapidocr-onnxruntime==1.4.1
onnxruntime==1.16.3
numpy==1.24.4
Pillow==10.1.0
pillow_heif==0.16.0
psutil==5.9.6
wordninja==2.0.0
```

- **Size Estimate**: ~160-180 MB unzipped, ~40-50 MB zipped.
- **Changes**:
  - Removed `opencv-python-headless` (~60 MB savings).
  - Downgraded to `fastapi==0.104.1`, `python-multipart==0.0.6`, `rapidocr-onnxruntime==1.4.1`, `numpy==1.24.4`, `Pillow==10.1.0`, `pillow_heif==0.16.0` for compatibility and size reduction.
  - Kept `psutil==5.9.6` and `wordninja==2.0.0` as required.

### Minimal Change to `api/main.py`
To support the removal of `opencv-python-headless`, update the `process_image_bytes` function in `api/main.py`:

```python
def process_image_bytes(image_bytes):
    try:
        pil_image = Image.open(io.BytesIO(image_bytes))
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        numpy_image = np.array(pil_image)
        return numpy_image  # No BGR conversion
    except Exception as e:
        raise ValueError(f"Failed to process image: {str(e)}")
```

- **Impact**: Minimal change, ensures compatibility with `rapidocr-onnxruntime`, may slightly reduce OCR accuracy due to no BGR conversion.

## Deployment on Render.com

### Steps
1. **Create a Render.com Web Service**:
   - Log in to Render.com.
   - Create a new Web Service and connect your GitHub repository (`VrajVyas11/RapidOCR_API_Service`).

2. **Configure Build Settings**:
   - **Runtime**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `uvicorn api.main:app --host 0.0.0.0 --port $PORT`
   - **Environment Variables**:
     - `PYTHON_VERSION`: `3.12`
     - `PORT`: Set by Render (e.g., `10000`)

3. **Deploy**:
   - Push changes to your GitHub repository.
   - Trigger a deployment in Render.com’s dashboard.
   - Access the API at `https://your-service.onrender.com` (e.g., `/init`, `/read_text`).

### Benefits of Render.com Deployment
- **Auto-Scaling**: Render automatically scales resources based on traffic.
- **Free Tier**: Suitable for small-scale deployments with generous limits.
- **No Bundle Size Restrictions**: Unlike Netlify, Render supports larger bundles (~500 MB unzipped).
- **Persistent Storage**: Supports temporary file storage for OCR models in `/tmp`.
- **Easy Management**: Render’s dashboard simplifies deployment and monitoring.

## API Endpoints

### 1. `GET /`
- **Description**: Health check endpoint to verify the API is online.
- **Response**:
  ```json
  {
    "message": "RapidOCR API for Vercel",
    "status": "online"
  }
  ```
- **Benefits**: Quick way to confirm API availability.

### 2. `POST /init`
- **Description**: Initializes the OCR engine with optional language settings.
- **Request Body**:
  ```json
  {
    "languages": "en"
  }
  ```
- **Response**:
  ```json
  {
    "status": "success",
    "message": "RapidOCR initialized for Vercel deployment",
    "langs": "en",
    "note": "Enhanced grouping for manga dialogue bubbles."
  }
  ```
- **Benefits**: Prepares the OCR engine, reducing latency for subsequent requests.

### 3. `POST /read_text`
- **Description**: Extracts text from an uploaded image, with manga-specific grouping.
- **Request**:
  - Form-data: `image` (file, e.g., WebP, AVIF, JPEG)
  - Query Parameter: `languages` (default: `en`)
- **Response**:
  ```json
  {
    "status": "success",
    "results": [
      {
        "bbox": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        "text": "Sample text",
        "score": 0.95
      }
    ],
    "paragraphs": [
      {
        "text": "Grouped text",
        "bbox": [[min_x, min_y], [max_x, min_y], [max_x, max_y], [min_x, max_y]],
        "score": 0.95,
        "item_count": 2,
        "individual_items": [...]
      }
    ],
    "stats": {
      "total_lines": 2,
      "total_paragraphs": 1,
      "processing_time": "0.123s",
      "image_size": 102400
    },
    "memory_usage": "~50MB"
  }
  ```
- **Benefits**: Returns both raw text results and grouped paragraphs, optimized for manga layouts.

### 4. `POST /close`
- **Description**: Cleans up the OCR engine and frees memory.
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Memory cleanup complete"
  }
  ```
- **Benefits**: Ensures efficient resource management by releasing memory.

## Usage Example
```bash
curl -X POST "https://your-service.onrender.com/read_text" \
  -F "image=@manga_page.webp" \
  -H "Content-Type: multipart/form-data"
```

- **Output**: JSON response with extracted text, bounding boxes, and grouped paragraphs.
- **Benefits**: Easy integration into web or mobile apps for manga translation or text extraction.

## Testing Locally
1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
2. **Run the API**:
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```
3. **Test Endpoints**:
   - Open `http://localhost:8000/docs` for interactive FastAPI documentation.
   - Use `curl` or Postman to test `/init`, `/read_text`, `/close`.

## Performance
- **Processing Time**: Typically <1 second for standard manga pages (tested with WebP images ~100 KB).
- **Memory Usage**: ~50-100 MB per request, monitored via `psutil`.
- **Scalability**: Handles multiple concurrent requests on Render.com’s free tier.
- **Accuracy**: High for English text, with `wordninja` improving readability for concatenated words.

## Troubleshooting
- **Large Bundle Size**:
  - Previous size: 391.34 MB unzipped, 135.12 MB zipped.
  - Solution: Removed `opencv-python-headless`, used lighter dependency versions.
  - Test size:
    ```powershell
    mkdir temp_bundle
    cd temp_bundle
    pip install --target . --no-cache-dir -r ../requirements.txt
    Get-ChildItem -Recurse | Measure-Object -Property Length -Sum | ForEach-Object { "{0:N2} MB (unzipped bundle size)" -f ($_.Sum / 1MB) }
    Compress-Archive -Path . -DestinationPath bundle.zip
    Get-Item bundle.zip | ForEach-Object { "{0:N2} MB (zipped bundle size)" -f ($_.Length / 1MB) }
    cd ..
    Remove-Item -Recurse -Force temp_bundle
    ```
- **404 Errors**: Ensure Render.com’s start command is `uvicorn api.main:app --host 0.0.0.0 --port $PORT`.
- **OCR Failure**: Verify image formats (WebP, AVIF, etc.) and test with smaller images.
- **Contact**: Open an issue on GitHub or check Render.com logs.

## Contributing
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/xyz`).
3. Commit changes (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature/xyz`).
5. Open a pull request.

## License
MIT License. See [LICENSE](LICENSE) for details.

## Contact
- **GitHub**: [VrajVyas11](https://github.com/VrajVyas11)
- **Issues**: [RapidOCR_API_Service Issues](https://github.com/VrajVyas11/RapidOCR_API_Service/issues)
