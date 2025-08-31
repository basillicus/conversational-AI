# AI Chat Lifecycle Demo

This repository is a compact, **one-project** demonstration of the full AI/ML lifecycle:
data preparation → model training → deployment → monitoring → retraining,
integrated into a conversational AI flow (local intent model + LLM via Ollama).

It is intentionally minimal and educational so you can:
- show a single coherent pipeline in interviews,
- run everything locally without cloud infra,
- demonstrate PO-level understanding of lifecycle, telemetry, and retraining.

## What is included
- `prepare_data.py` - create a small synthetic intent dataset
- `train_model.py` - trains a TF-IDF + LogisticRegression intent pipeline and saves it
- `main.py` - FastAPI app that exposes `/chat`, `/predict_intent`, `/retrain`, `/metrics`, `/submit_feedback`
- `gradio_app.py` - small frontend to test the system
- `monitoring.py` - parse logs and produce a simple metrics summary
- `requirements.txt` - Python dependencies
- `docs/` - MkDocs-ready documentation (see below)

## Quickstart (local)
1. Create virtualenv and install:
   ```
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```
2. Prepare data and train:
   ```
   python prepare_data.py
   python train_model.py
   ```
   This writes `models/intent_pipeline.joblib`.

3. Run the FastAPI app:
   ```
   uvicorn main:app --reload
   ```
   The app will be available at `http://127.0.0.1:8000`.

4. (Optional) Run the Gradio UI in another terminal:
   ```
   python gradio_app.py
   ```
   Open `http://127.0.0.1:7860` to interact.

## Using Ollama
If you have Ollama installed and a model served, set the environment variable:
```
export OLLAMA_API_URL=http://localhost:11434
```
Then call the `/chat` endpoint with `mode=llm` or supply `ab_prob` to let the system decide.

The code automatically falls back to a fake LLM if Ollama is not reachable.

## Retraining (demo)
- Users can send feedback/examples via `/submit_feedback`.
- Trigger retraining with:
  ```
  curl -X POST http://127.0.0.1:8000/retrain
  ```
  This runs the same simple training pipeline and reloads the model in the running app.

**Note:** in production, retraining should be an asynchronous pipeline with validation, approval gates, CI/CD and model registry. This demo runs it inline for clarity and speed.

## Monitoring & Anomalies
- Each chat request writes a JSON line into `logs/chat_metrics.log`.
- `/metrics` parses the log and returns simple aggregates.
- Anomalies are flagged by simple rules (low confidence, suspicious tokens, long prompt).
- Use `python monitoring.py` to produce a CSV summary.

## Documentation (MkDocs)
A basic MkDocs site is under `docs/`. To preview:
```
pip install mkdocs
mkdocs serve
```
Then open `http://127.0.0.1:8000` (MkDocs default is 8000) — change port if needed.

## Files produced
After running the training step you will see:
- `models/intent_pipeline.joblib` (the saved pipeline)
- `models/model_metadata.json`
- `logs/chat_metrics.log` (chat telemetry)
- `data/new_user_data.csv` (collected feedback)

## Appendix: How this maps to the job description
- **Data preparation:** `prepare_data.py` + `data/`
- **Model development:** `train_model.py` (experiments, metrics)
- **Deployment:** `main.py` (FastAPI endpoints serving model and LLM)
- **Monitoring:** `logs/` + `/metrics` endpoint + `monitoring.py`
- **Experimentation:** `ab_prob` parameter drives simple A/B logic between local and LLM
- **Retraining / CI/CD stub:** `/submit_feedback` + `/retrain` (in-process retrain)

---
This demo is intentionally compact. The goal is to **demonstrate my understanding of the lifecycle and can produce working artifacts**, not to replace production-grade MLOps.
