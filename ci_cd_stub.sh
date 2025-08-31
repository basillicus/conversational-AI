#!/usr/bin/env bash
# ci_cd_stub.sh - simple idea of how retrain -> test -> deploy could be automated
# In CI: run tests, then call train_model.py to produce artifacts, validate them, then publish model to artifact storage.
set -e
python prepare_data.py
python train_model.py
# Run a smoke test against the local FastAPI app (expected to be running)
curl -s -X POST http://127.0.0.1:8000/predict_intent -d 'prompt=hello' -H 'Content-Type: application/x-www-form-urlencoded'
echo "CI/CD stub completed."
