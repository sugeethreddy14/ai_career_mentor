#!/usr/bin/env bash
# Simple start script used by Render for the backend
set -e
# Prefer lightweight requirements for Render if present
REQ_FILE="requirements_render.txt"
if [ ! -f "$REQ_FILE" ]; then
	REQ_FILE="requirements.txt"
fi
echo "Installing from $REQ_FILE"
pip install -r "$REQ_FILE"

echo "Starting backend on port 10000"
uvicorn app.main:app --host 0.0.0.0 --port 10000