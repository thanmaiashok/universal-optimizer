#!/bin/bash
# PredycatAI Universal Optimizer - Kill Script
# Stops all PredycatAI processes

echo "🛑 Stopping PredycatAI..."

# Kill by PID
if [ -f .predycat_pid ]; then
    PID=$(cat .predycat_pid)
    if kill -0 $PID 2>/dev/null; then
        kill $PID
        echo "✅ Stopped API (PID: $PID)"
    fi
    rm .predycat_pid
fi

# Kill any remaining
pkill -f "predycat_ai" 2>/dev/null
pkill -f "python3.*api" 2>/dev/null

echo "✅ PredycatAI stopped"