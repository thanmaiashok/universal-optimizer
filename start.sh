#!/bin/bash
# OptiLLM Universal Optimizer - Start Script
# Starts the API server and CLI tools

echo "🚀 Starting OptiLLM Universal Optimizer..."

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found. Install from python.org"
    exit 1
fi

echo "Python: $(python3 --version)"

# Activate venv if present
if [ -f "venv/bin/activate" ]; then
    source venv/bin/activate
fi

# Install dependencies if needed
if ! python3 -c "import fastapi" 2>/dev/null; then
    echo "📦 Installing dependencies..."
    pip3 install torch numpy transformers fastapi uvicorn pydantic psutil
fi

# Load HF token from .env if present
if [ -f ".env" ]; then
    export $(grep -v '^#' .env | xargs)
fi

# Create directories
mkdir -p outputs/models outputs/reports outputs/logs models cache

# Kill all old optillm processes and anything on port 8080
pkill -f "opti_llm" 2>/dev/null || true
pkill -f "uvicorn optillm" 2>/dev/null || true
lsof -ti :8080 | xargs kill -9 2>/dev/null || true
sleep 1

# Start API server in background
echo "🌐 Starting API server on port 8080..."
uvicorn opti_llm.api.main:app --host 0.0.0.0 --port 8080 &
API_PID=$!

echo "✅ OptiLLM running!"
echo "   API:       http://localhost:8080"
echo "   Docs:      http://localhost:8080/docs"
echo "   Dashboard: http://localhost:8080/dashboard"
echo ""
echo "Press Ctrl+C to stop"
echo "API PID: $API_PID"

# Save PID for kill script
echo $API_PID > .optillm_pid

wait