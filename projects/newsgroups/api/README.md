# 20 Newsgroups Classification API

This FastAPI application provides real-time newsgroup classification using trained machine learning models.

## 🚀 Quick Start

### Start the API Server
```bash
cd api
./start_api.sh
```

The API will be available at:
- **API Base URL**: http://localhost:8000
- **Interactive Docs**: http://localhost:8000/docs
- **ReDoc Documentation**: http://localhost:8000/redoc

### Test the API
```bash
# Run demo
python demo_api.py

# Run comprehensive tests
python test_api.py
```

## 📡 API Endpoints

### Core Endpoints

#### `GET /`
Root endpoint with API information.

#### `GET /health`
Health check endpoint.
```json
{
  "status": "healthy",
  "available_models": ["naive_bayes", "logistic_regression", ...],
  "loaded_models": ["naive_bayes"]
}
```

#### `GET /models`
List all available models and their status.
```json
[
  {
    "model_name": "naive_bayes",
    "is_loaded": true,
    "categories": ["alt.atheism", "comp.graphics", ...]
  }
]
```

### Prediction Endpoints

#### `POST /predict`
Classify a single text.
```json
{
  "text": "Looking for advice on computer graphics programming",
  "model": "naive_bayes"  // optional
}
```

Response:
```json
{
  "predicted_category": "comp.graphics",
  "confidence": 0.85,
  "model_used": "naive_bayes",
  "all_predictions": {
    "comp.graphics": 0.85,
    "comp.os.ms-windows.misc": 0.10,
    ...
  }
}
```

#### `POST /predict/batch`
Classify multiple texts (up to 100).
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "model": "naive_bayes"  // optional
}
```

### Model Management

#### `POST /models/{model_name}/load`
Load a specific model.
```bash
curl -X POST "http://localhost:8000/models/naive_bayes/load"
```

#### `DELETE /models/{model_name}/unload`
Unload a specific model.
```bash
curl -X DELETE "http://localhost:8000/models/naive_bayes/unload"
```

#### `POST /models/load-all`
Load all available models.
```bash
curl -X POST "http://localhost:8000/models/load-all"
```

## 🔧 Configuration

### Environment Variables
- `API_HOST`: Host address (default: 0.0.0.0)
- `API_PORT`: Port number (default: 8000)
- `LOG_LEVEL`: Logging level (default: info)

### Model Configuration
Models are automatically loaded from the `results/models/` directory. Each model requires:
- `{model_name}_model.pkl`: Trained model
- `{model_name}_vectorizer.pkl`: Text vectorizer
- `{model_name}_label_encoder.pkl`: Label encoder

## 📊 Usage Examples

### Python Client
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={"text": "Computer graphics programming help"}
)
result = response.json()
print(f"Category: {result['predicted_category']}")
print(f"Confidence: {result['confidence']:.3f}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={"texts": ["Text 1", "Text 2"]}
)
results = response.json()["predictions"]
```

### cURL Examples
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "Looking for advice on computer graphics programming"}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Text 1", "Text 2", "Text 3"]}'

# Health check
curl -X GET "http://localhost:8000/health"
```

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
uvicorn newsgroups_api:app --reload --host 0.0.0.0 --port 8000
```

### Testing
```bash
# Run tests
python test_api.py

# Run demo
python demo_api.py
```

## 📈 Performance

### Expected Response Times
- Single prediction: 50-200ms
- Batch prediction (10 texts): 200-500ms
- Model loading: 1-5 seconds

### Memory Usage
- Base API: ~100MB
- Per loaded model: ~50-200MB (depending on model size)

## 🔒 Security Considerations

- API accepts text input up to 10,000 characters
- Batch requests limited to 100 texts
- CORS enabled for all origins (configure for production)
- No authentication implemented (add for production use)

## 🐛 Troubleshooting

### Common Issues

1. **Model not found**
   ```
   Error: Model files not found for naive_bayes
   ```
   **Solution**: Ensure models are trained and saved in `results/models/`

2. **API not responding**
   ```
   Connection refused
   ```
   **Solution**: Check if API server is running on port 8000

3. **Import errors**
   ```
   ModuleNotFoundError: No module named 'sklearn'
   ```
   **Solution**: Install requirements: `pip install -r requirements.txt`

### Debug Mode
```bash
# Run with debug logging
uvicorn newsgroups_api:app --log-level debug
```

## 📚 API Documentation

Visit http://localhost:8000/docs for interactive API documentation with:
- Endpoint descriptions
- Request/response schemas
- Try-it-out functionality
- Authentication details (if implemented)

---

*For more information, see the main project README.md*
