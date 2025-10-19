# GoEmotions Multilabel Classification API

This FastAPI application provides real-time emotion analysis using trained machine learning models.

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
    "emotion_categories": ["joy", "sadness", "anger", ...]
  }
]
```

### Prediction Endpoints

#### `POST /predict`
Analyze emotions in a single text.
```json
{
  "text": "I'm so excited and happy about this news!",
  "model": "naive_bayes",
  "threshold": 0.5
}
```

Response:
```json
{
  "text": "I'm so excited and happy about this news!",
  "predicted_emotions": [
    {
      "emotion": "joy",
      "confidence": 0.85,
      "predicted": true
    },
    {
      "emotion": "excitement",
      "confidence": 0.72,
      "predicted": true
    },
    {
      "emotion": "sadness",
      "confidence": 0.12,
      "predicted": false
    }
  ],
  "model_used": "naive_bayes",
  "threshold": 0.5,
  "total_emotions": 27,
  "predicted_count": 2
}
```

#### `POST /predict/batch`
Analyze emotions in multiple texts (up to 100).
```json
{
  "texts": ["Text 1", "Text 2", "Text 3"],
  "model": "naive_bayes",
  "threshold": 0.4
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
- `{model_name}_emotion_categories.pkl`: Emotion categories list

## 📊 Usage Examples

### Python Client
```python
import requests

# Single prediction
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "text": "I'm feeling great today!",
        "threshold": 0.4
    }
)
result = response.json()

# Show predicted emotions
for emotion in result['predicted_emotions']:
    if emotion['predicted']:
        print(f"{emotion['emotion']}: {emotion['confidence']:.3f}")

# Batch prediction
response = requests.post(
    "http://localhost:8000/predict/batch",
    json={
        "texts": ["I'm happy!", "This is sad.", "I'm excited!"],
        "threshold": 0.3
    }
)
results = response.json()["predictions"]
```

### cURL Examples
```bash
# Single prediction
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"text": "I am so excited about this!", "threshold": 0.5}'

# Batch prediction
curl -X POST "http://localhost:8000/predict/batch" \
     -H "Content-Type: application/json" \
     -d '{"texts": ["Happy text", "Sad text"], "threshold": 0.4}'

# Health check
curl -X GET "http://localhost:8000/health"
```

## 🎯 Emotion Categories

The API can detect 27 different emotions:

**Positive Emotions:**
- `admiration`, `amusement`, `approval`, `caring`, `excitement`, `gratitude`, `joy`, `love`, `optimism`, `pride`, `relief`

**Negative Emotions:**
- `anger`, `annoyance`, `disappointment`, `disapproval`, `disgust`, `embarrassment`, `fear`, `grief`, `nervousness`, `remorse`, `sadness`

**Neutral/Other:**
- `confusion`, `curiosity`, `desire`, `realization`, `surprise`, `neutral`

## 🔍 Threshold Tuning

The `threshold` parameter controls how confident the model needs to be to predict an emotion:

- **Low threshold (0.1-0.3)**: More emotions predicted, higher recall
- **Medium threshold (0.4-0.6)**: Balanced predictions
- **High threshold (0.7-0.9)**: Fewer emotions predicted, higher precision

### Example Threshold Effects
```python
# Low threshold - more emotions detected
{"text": "I'm happy!", "threshold": 0.2}
# Might predict: joy, excitement, optimism

# High threshold - only confident predictions
{"text": "I'm happy!", "threshold": 0.8}
# Might predict: joy only
```

## 🛠️ Development

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run in development mode
uvicorn emotion_api:app --reload --host 0.0.0.0 --port 8000
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

4. **No emotions predicted**
   ```
   predicted_count: 0
   ```
   **Solution**: Try lowering the threshold parameter

### Debug Mode
```bash
# Run with debug logging
uvicorn emotion_api:app --log-level debug
```

## 📚 API Documentation

Visit http://localhost:8000/docs for interactive API documentation with:
- Endpoint descriptions
- Request/response schemas
- Try-it-out functionality
- Authentication details (if implemented)

---

*For more information, see the main project README.md*
