#!/bin/bash

PROJECT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  LungSeg AI — Starting up"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# ── Check venv exists ──────────────────────
if [ ! -d "$PROJECT_DIR/venv" ]; then
  echo "❌  Virtual environment not found."

  
  echo ""
  echo "    Run this first:"
  echo "    python -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  echo ""
  exit 1
fi

# ── Check model weights exist ──────────────
if [ ! -f "$PROJECT_DIR/checkpoints/best_model.pth" ]; then
  echo "❌  Model weights not found at checkpoints/best_model.pth"
  echo ""
  echo "    The API cannot start without the trained model."
  echo "    Ask the author for best_model.pth and place it at:"
  echo "    $PROJECT_DIR/checkpoints/best_model.pth"
  echo ""
  exit 1
fi

# ── Activate venv ──────────────────────────
source "$PROJECT_DIR/venv/bin/activate"

# ── Start API ─────────────────────────────
echo "→  Starting API on http://127.0.0.1:8000 ..."
cd "$PROJECT_DIR"
uvicorn api.main:app --host 127.0.0.1 --port 8000 &
API_PID=$!

# ── Start frontend server ─────────────────
echo "→  Starting frontend on http://127.0.0.1:3000 ..."
cd "$PROJECT_DIR/frontend"
python -m http.server 3000 --bind 127.0.0.1 &
FRONTEND_PID=$!

# ── Wait for API to be ready ──────────────
echo "→  Waiting for API to load model..."
for i in $(seq 1 30); do
  sleep 1
  if curl -s http://127.0.0.1:8000/health > /dev/null 2>&1; then
    break
  fi
done

# ── Open browser ──────────────────────────
echo "→  Opening browser..."
sleep 1
open http://127.0.0.1:3000/index.html 2>/dev/null || \
xdg-open http://127.0.0.1:3000/index.html 2>/dev/null || \
echo "   Open http://127.0.0.1:3000/index.html in your browser"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅  LungSeg AI is running"
echo ""
echo "  Landing page : http://127.0.0.1:3000/index.html"
echo "  Tool         : http://127.0.0.1:3000/tool.html"
echo "  API docs     : http://127.0.0.1:8000/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Press Ctrl+C to stop everything."
echo ""

# ── Clean up on exit ──────────────────────
trap "echo ''; echo 'Shutting down...'; kill $API_PID $FRONTEND_PID 2>/dev/null; echo 'Stopped.'; exit 0" INT TERM
wait
