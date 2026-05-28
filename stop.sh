#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

if [ -f logs/nvidia_exporter.pid ]; then
    PID=$(cat logs/nvidia_exporter.pid)
    if kill -0 "$PID" 2>/dev/null; then
        kill "$PID"
        rm logs/nvidia_exporter.pid
        echo "Nvidia exporter stopped (PID: $PID)"
    else
        echo "Process not running"
        rm -f logs/nvidia_exporter.pid
    fi
else
    echo "PID file not found"
fi
