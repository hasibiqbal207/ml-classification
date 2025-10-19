#!/bin/bash
# SMS Spam Detection API Startup Script

set -e

echo "🚀 Starting SMS Spam Detection API..."

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "⚠️  Warning: No virtual environment detected"
    echo "   Consider activating your environment: source /home/hasib/tfenv/bin/activate"
fi

# Check if required packages are installed
echo "📦 Checking dependencies..."
python -c "import fastapi, uvicorn, sklearn, numpy, pandas" 2>/dev/null || {
    echo "❌ Missing dependencies. Installing..."
    pip install -r requirements.txt
}

# Check if models exist
echo "🤖 Checking for trained models..."
if [ ! -d "../results" ]; then
    echo "❌ No results directory found. Please train models first:"
    echo "   cd .. && python scripts/train_naive_bayes.py"
    echo "   cd .. && python scripts/train_logistic_regression.py --no-tuning"
    echo "   cd .. && python scripts/train_random_forest.py --no-tuning"
    exit 1
fi

MODEL_COUNT=$(ls ../results/*_model.pkl 2>/dev/null | wc -l)
if [ $MODEL_COUNT -eq 0 ]; then
    echo "❌ No trained models found. Please train models first."
    exit 1
fi

echo "✅ Found $MODEL_COUNT trained models"

# Create logs directory
mkdir -p logs

# Set environment variables
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export LOG_LEVEL="INFO"

# Start the API server
echo "🌐 Starting API server on http://localhost:8000"
echo "📚 API Documentation: http://localhost:8000/docs"
echo "🔍 Health Check: http://localhost:8000/health"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run the API
uvicorn sms_api:app --host 0.0.0.0 --port 8000 --reload --log-level info
