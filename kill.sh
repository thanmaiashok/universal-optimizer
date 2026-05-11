#!/bin/bash
# OptiLLM Universal Optimizer - Kill Script
# Stops all OptiLLM processes

echo "🛑 Stopping OptiLLM..."

# Kill by PID
if [ -f .optillm_pid ]; then
    PID=$(cat .optillm_pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ Stopped API (PID: $PID)"
    fi
    rm .optillm_pid
fi

# Kill any remaining
pkill -f "opti_llm" 2>/dev/null
pkill -f "python3.*api" 2>/dev/null

echo "✅ OptiLLM stopped"