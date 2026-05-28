#!/bin/bash
# Start nvidia exporter
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

mkdir -p logs

# Check if already running
if [ -f logs/nvidia_exporter.pid ]; then
    PID=$(cat logs/nvidia_exporter.pid)
    if kill -0 "$PID" 2>/dev/null; then
        echo "Nvidia exporter already running (PID: $PID)"
        exit 0
    fi
fi

nohup python3 nvidia_exporter.py > logs/stdout.log 2>&1 &
echo $! > logs/nvidia_exporter.pid
echo "Nvidia exporter started (PID: $!)"
echo "Metrics available at: http://localhost:${NVIDIA_EXPORTER_PORT:-9835}/metrics"
