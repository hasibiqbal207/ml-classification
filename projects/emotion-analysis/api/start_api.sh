#!/bin/bash

# Start script for GoEmotions Multilabel Classification API

echo "Starting GoEmotions Multilabel Classification API..."

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Start the API server
echo "Starting API server on http://localhost:8000"
echo "API documentation available at http://localhost:8000/docs"
echo "Press Ctrl+C to stop the server"

python emotion_api.py
