#!/usr/bin/env bash
# Simple start script used by Render for the frontend (Streamlit)
set -e
# Prefer lightweight requirements for Render if present
REQ_FILE="requirements_render.txt"
if [ ! -f "$REQ_FILE" ]; then
	REQ_FILE="requirements.txt"
fi
echo "Installing from $REQ_FILE"
pip install -r "$REQ_FILE"

echo "Starting Streamlit on port 10000"
streamlit run ui/app.py --server.port=10000 --server.address=0.0.0.0