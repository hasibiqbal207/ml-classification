# SMS Spam Detection API Documentation

## 🚀 Overview

The SMS Spam Detection API is a real-time machine learning service that classifies SMS messages as spam or ham (legitimate). It provides RESTful endpoints for single message classification, batch processing, and ensemble predictions using multiple trained models.

## 📋 Features

- **Real-time Classification**: Instant spam/ham detection
- **Multiple Models**: Naive Bayes, Logistic Regression, Random Forest
- **Ensemble Predictions**: Combines multiple models for better accuracy
- **Batch Processing**: Process multiple messages in a single request
- **High Performance**: FastAPI-based with async support
- **Comprehensive Testing**: Built-in test suite and benchmarks
- **Production Ready**: Docker support and monitoring

## 🛠 Installation & Setup

### Prerequisites
- Python 3.10+
- Trained models (run training scripts first)
- Virtual environment (recommended)

### Quick Start

1. **Activate Environment**:
   ```bash
   source /home/hasib/tfenv/bin/activate
   ```

2. **Install Dependencies**:
   ```bash
   cd projects/binary/api
   pip install -r requirements.txt
   ```

3. **Start API Server**:
   ```bash
   ./start_api.sh
   ```

4. **Access API Documentation**:
   - Open http://localhost:8000/docs
   - Interactive API documentation with Swagger UI

## 📡 API Endpoints

### Base URL
```
http://localhost:8000
```

### 1. Health Check
```http
GET /health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-10-19T02:45:00",
  "models_loaded": 3,
  "total_models": 3,
  "uptime_seconds": 3600
}
```

### 2. List Available Models
```http
GET /models
```

**Response**:
```json
[
  {
    "model_name": "naive_bayes",
    "accuracy": 0.9869,
    "precision": 0.9636,
    "recall": 0.9381,
    "f1_score": 0.9507,
    "roc_auc": 0.9911,
    "is_loaded": true,
    "last_updated": "2025-10-19T02:45:00"
  }
]
```

### 3. Single Message Prediction
```http
POST /predict
```

**Request Body**:
```json
{
  "text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
  "sender_id": "optional_sender_id",
  "timestamp": "2025-10-19T02:45:00"
}
```

**Query Parameters**:
- `model`: Model to use (`naive_bayes`, `logistic_regression`, `random_forest`)

**Response**:
```json
{
  "text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
  "prediction": "spam",
  "probability": 0.9876,
  "model_used": "naive_bayes",
  "processing_time_ms": 15.2,
  "sender_id": "optional_sender_id",
  "timestamp": "2025-10-19T02:45:00"
}
```

### 4. Batch Prediction
```http
POST /predict/batch
```

**Request Body**:
```json
{
  "messages": [
    {
      "text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
      "sender_id": "sender1"
    },
    {
      "text": "Go until jurong point, crazy.. Available only in bugis",
      "sender_id": "sender2"
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [
    {
      "text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
      "prediction": "spam",
      "probability": 0.9876,
      "model_used": "naive_bayes",
      "processing_time_ms": 12.1,
      "sender_id": "sender1"
    }
  ],
  "total_messages": 2,
  "total_processing_time_ms": 25.3,
  "spam_count": 1,
  "ham_count": 1
}
```

### 5. Ensemble Prediction
```http
POST /predict/ensemble
```

**Request Body**:
```json
{
  "text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
  "sender_id": "test_sender"
}
```

**Response**:
```json
{
  "text": "Free entry in 2 a wkly comp to win FA Cup final tkts",
  "prediction": "spam",
  "probability": 0.9789,
  "model_used": "ensemble_3_models",
  "processing_time_ms": 45.6,
  "sender_id": "test_sender"
}
```

### 6. API Statistics
```http
GET /stats
```

**Response**:
```json
{
  "models_loaded": 3,
  "available_models": ["naive_bayes", "logistic_regression", "random_forest"],
  "uptime_seconds": 3600,
  "api_version": "1.0.0"
}
```

## 🧪 Testing

### Run Comprehensive Tests
```bash
python test_api.py
```

### Test Specific Endpoints
```bash
# Test with custom URL
python test_api.py --url http://localhost:8000

# Test specific model
python test_api.py --model naive_bayes

# Run performance benchmark
python test_api.py --benchmark 100
```

### Manual Testing with curl

1. **Health Check**:
   ```bash
   curl http://localhost:8000/health
   ```

2. **Single Prediction**:
   ```bash
   curl -X POST "http://localhost:8000/predict?model=naive_bayes" \
        -H "Content-Type: application/json" \
        -d '{"text": "Free entry in 2 a wkly comp to win FA Cup final tkts"}'
   ```

3. **Batch Prediction**:
   ```bash
   curl -X POST "http://localhost:8000/predict/batch?model=naive_bayes" \
        -H "Content-Type: application/json" \
        -d '{"messages": [{"text": "Free entry in 2 a wkly comp"}, {"text": "Hello how are you"}]}'
   ```

## 🐳 Docker Deployment

### Build and Run with Docker
```bash
# Build Docker image
docker build -t sms-spam-api .

# Run container
docker run -p 8000:8000 -v $(pwd)/../results:/app/results:ro sms-spam-api
```

### Docker Compose (Recommended)
```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f sms-api

# Stop services
docker-compose down
```

## 📊 Performance Benchmarks

### Typical Performance Metrics
- **Response Time**: 10-50ms per prediction
- **Throughput**: 100-500 requests/second
- **Memory Usage**: ~200MB (with all models loaded)
- **Accuracy**: 98.69% (Naive Bayes), 97.37% (Logistic Regression), 97.73% (Random Forest)

### Load Testing
```bash
# Install Apache Bench
sudo apt-get install apache2-utils

# Run load test
ab -n 1000 -c 10 -H "Content-Type: application/json" \
   -p test_message.json http://localhost:8000/predict
```

## 🔧 Configuration

### Environment Variables
- `PYTHONPATH`: Python path for imports
- `LOG_LEVEL`: Logging level (DEBUG, INFO, WARNING, ERROR)
- `API_HOST`: Host to bind to (default: 0.0.0.0)
- `API_PORT`: Port to bind to (default: 8000)

### Model Configuration
Models are automatically loaded from the `../results/` directory. Ensure the following files exist:
- `naive_bayes_model.pkl`
- `naive_bayes_vectorizer.pkl`
- `logistic_regression_model.pkl`
- `logistic_regression_vectorizer.pkl`
- `random_forest_model.pkl`
- `random_forest_vectorizer.pkl`

## 🚨 Error Handling

### Common Error Responses

**400 Bad Request**:
```json
{
  "detail": "Model 'invalid_model' not available",
  "timestamp": "2025-10-19T02:45:00"
}
```

**422 Validation Error**:
```json
{
  "detail": [
    {
      "loc": ["body", "text"],
      "msg": "field required",
      "type": "value_error.missing"
    }
  ]
}
```

**500 Internal Server Error**:
```json
{
  "detail": "Prediction failed: Model not loaded",
  "timestamp": "2025-10-19T02:45:00"
}
```

## 🔒 Security Considerations

### Production Deployment
1. **Authentication**: Add JWT token authentication
2. **Rate Limiting**: Implement request rate limiting
3. **CORS**: Configure appropriate CORS policies
4. **HTTPS**: Use SSL/TLS certificates
5. **Input Validation**: Sanitize all inputs
6. **Logging**: Monitor and log all requests

### Example Security Headers
```python
# Add to FastAPI app
from fastapi.middleware.trustedhost import TrustedHostMiddleware

app.add_middleware(
    TrustedHostMiddleware, 
    allowed_hosts=["yourdomain.com", "*.yourdomain.com"]
)
```

## 📈 Monitoring & Logging

### Log Files
- `sms_api.log`: Application logs
- `logs/`: Directory for structured logs

### Metrics Collection
The API includes Prometheus metrics for:
- Request count and duration
- Model prediction accuracy
- Error rates
- System resource usage

### Health Monitoring
```bash
# Check API health
curl http://localhost:8000/health

# Monitor logs
tail -f sms_api.log
```

## 🔄 Integration Examples

### Python Client
```python
import requests

def classify_sms(text, model='naive_bayes'):
    response = requests.post(
        'http://localhost:8000/predict',
        params={'model': model},
        json={'text': text}
    )
    return response.json()

# Usage
result = classify_sms("Free entry in 2 a wkly comp")
print(f"Prediction: {result['prediction']}")
print(f"Confidence: {result['probability']:.2%}")
```

### JavaScript Client
```javascript
async function classifySMS(text, model = 'naive_bayes') {
    const response = await fetch(`http://localhost:8000/predict?model=${model}`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ text: text })
    });
    return await response.json();
}

// Usage
classifySMS("Free entry in 2 a wkly comp")
    .then(result => console.log(result));
```

## 🆘 Troubleshooting

### Common Issues

1. **Models Not Loading**:
   - Ensure trained models exist in `../results/`
   - Check file permissions
   - Verify model files are not corrupted

2. **Import Errors**:
   - Activate virtual environment
   - Install all dependencies: `pip install -r requirements.txt`
   - Check Python path

3. **Performance Issues**:
   - Monitor memory usage
   - Check CPU utilization
   - Consider model optimization

4. **Connection Errors**:
   - Verify API is running on correct port
   - Check firewall settings
   - Ensure no port conflicts

### Debug Mode
```bash
# Run with debug logging
LOG_LEVEL=DEBUG uvicorn sms_api:app --reload
```

## 📚 Additional Resources

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Uvicorn Documentation](https://www.uvicorn.org/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)
- [Docker Documentation](https://docs.docker.com/)

---

*Last updated: October 19, 2025*
