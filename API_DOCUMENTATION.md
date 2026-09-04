# Lung Cancer Detection Backend API Documentation

## API Overview

Base URL: http://localhost:5000

### Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | /api/health | Check backend and model status |
| POST | /api/predict | Predict a single image |
| POST | /api/predict-pdf | Process a PDF containing text, images, or both |

## 1. Health Check

GET /api/health

Example:

```bash
curl http://localhost:5000/api/health
```

Expected response:

```json
{
  "status": "ok",
  "model_loaded": true,
  "image_model": "lung_cancer_model_v5.keras",
  "image_threshold": 0.5,
  "text_model": "tfidf_logistic_model.joblib",
  "text_threshold": 0.63
}
```

## 2. Image Prediction

POST /api/predict

Content-Type: multipart/form-data

Form field: image

Example:

```bash
curl -X POST -F "image=@$HOME/Desktop/your_image.jpg" http://localhost:5000/api/predict
```

Response:

```json
{
  "success": true,
  "type": "image",
  "model": "lung_cancer_model_v5.keras",
  "result": {
    "malignant_probability": 0.1428,
    "model_score": 0.8572,
    "non_malignant_probability": 0.8572,
    "prediction": "NON-MALIGNANT",
    "threshold": 0.5,
    "uncertainty": false
  }
}
```

React should use `data.result.prediction`, `data.result.malignant_probability`, `data.result.non_malignant_probability`, `data.result.model_score`, and `data.result.uncertainty`.

## 3. PDF Prediction

POST /api/predict-pdf

Content-Type: multipart/form-data

Form field: pdf

Example:

```bash
curl -X POST -F "pdf=@$HOME/Desktop/sample.pdf" http://localhost:5000/api/predict-pdf
```

The backend supports text PDFs, image PDFs, multi-page PDFs, and mixed PDFs.

Possible document types:

- `text`
- `image`
- `mixed`
- `unknown`

## 4. Text Results

The NLP model detects parenchymal lesions in radiology report text.

Example:

```json
{
  "prediction": "NO_PARENCHYMAL_LESION",
  "probability": 0.1034,
  "threshold": 0.63
}
```

Important: the text model does NOT diagnose cancer. React should use wording such as `Parenchymal lesion detected` or `No parenchymal lesion detected`.

## 5. Image Results From PDFs

Each processed image appears in `image_results` with:

- `page`
- `image_index`
- `result`

The `result` object has the same fields as `/api/predict`.

## 6. PDF Summary

Image summary:

```json
{
  "prediction": "NON-MALIGNANT",
  "average_malignant_probability": 0.1428,
  "maximum_malignant_probability": 0.1428,
  "images_processed": 2,
  "threshold": 0.5
}
```

Text summary:

```json
{
  "prediction": "NO_PARENCHYMAL_LESION",
  "average_probability": 0.1034,
  "pages_with_text_prediction": 1,
  "threshold": 0.63
}
```

## 7. Mixed PDF

A mixed PDF can contain image and text pages. The response contains `document_type`, `summary`, `text_results`, `image_results`, `pages_processed`, `text_pages`, and `image_pages`.

Example flow:

```text
Page 1 -> Image -> V5 model
Page 2 -> Text -> NLP model
Page 3 -> Image -> V5 model
```

## 8. Error Handling

Frontend should always check both HTTP status and `success`.

Possible errors include:

```json
{"success": false, "error": "No image file provided"}
```

```json
{"success": false, "error": "No PDF file provided"}
```

For unsupported files, invalid PDFs, oversized files, or unavailable backend, display a clear user-friendly message.

Maximum request size: 10 MB.

## 9. React Integration

Image upload:

```javascript
const formData = new FormData();
formData.append("image", selectedFile);

const response = await fetch("http://localhost:5000/api/predict", {
  method: "POST",
  body: formData
});

const data = await response.json();
```

PDF upload:

```javascript
const formData = new FormData();
formData.append("pdf", selectedFile);

const response = await fetch("http://localhost:5000/api/predict-pdf", {
  method: "POST",
  body: formData
});

const data = await response.json();
```

Do NOT manually set `Content-Type` when using `FormData`.

Do NOT send uploaded files using JSON.

Do NOT recreate model thresholds inside React. The backend owns the prediction logic.

## 10. Models

Image model:

`lung_cancer_model_v5.keras`

- MobileNetV2 based
- 224 x 224 RGB input
- CT nodule-crop training data
- Image threshold: 0.50

Text model:

`tfidf_logistic_model.joblib`

Used with `tfidf_vectorizer.joblib`.

- Detects parenchymal lesions in radiology report text
- Text threshold: 0.63

## 11. Testing Checklist

- [ ] Health endpoint works
- [ ] JPG upload works
- [ ] JPEG upload works
- [ ] PNG upload works
- [ ] Text PDF works
- [ ] Image PDF works
- [ ] Multi-page PDF works
- [ ] Mixed PDF works
- [ ] Errors are displayed correctly
- [ ] Backend unavailable state is handled

## 12. Responsibility Split

Backend team:

- Flask API
- Model loading
- Image prediction
- PDF processing
- Text extraction
- Text classification
- Image extraction
- Error handling

React team:

- File upload UI
- API calls
- Loading states
- Error messages
- Display image results
- Display text results
- Display PDF summaries
- Frontend styling

Shared:

- Endpoint names
- Form field names
- JSON response structure
- Error handling
- Backend URL

## Important Project Disclaimer

This is a college research/prototype system and is not a clinical diagnostic system. The image model is intended for the trained CT nodule-crop domain, and the text model detects parenchymal lesions rather than directly diagnosing cancer.
