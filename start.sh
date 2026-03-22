#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo "Starting LungSeg AI..."

# Start API in background
cd "$PROJECT_DIR"
source venv/bin/activate
uvicorn api.main:app --host localhost --port 8000 &
API_PID=$!
echo "API started (PID $API_PID)"

# Start frontend server in background
cd "$PROJECT_DIR/frontend"
python -m http.server 3000 &
FRONTEND_PID=$!
echo "Frontend started (PID $FRONTEND_PID)"

# Open browser
sleep 2
open http://localhost:3000/index.html

echo ""
echo "LungSeg AI is running."
echo "  Landing page: http://localhost:3000/index.html"
echo "  API:          http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop everything."

# Wait and clean up on exit
trap "kill $API_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'" EXIT
wait
