# API Reference

`POST /chat`
- body: `{ "prompt": "...", "mode": "auto|local|llm", "ab_prob": 0.0, "llm_model": "mistral" }`
- returns: `{ answer, chosen_model, latency_ms, intent, confidence, anomaly, anomaly_reason }`

`POST /predict_intent`
- form: `prompt=<text>`
- returns: `{ intent, confidence }`

`POST /submit_feedback`
- form: `text=<example>&label=<intent or empty>`
- appends to `data/new_user_data.csv`

`POST /retrain`
- triggers training and reloads the model into the running app (demo-only).

`GET /metrics`
- returns aggregated telemetry from `logs/chat_metrics.log`.
