# Notes, limitations and interview talking points

- **Not production-ready**: This demo purposefully runs retraining in-process and uses local files for artifacts.
- **How to extend to production**:
  - Move training to a dedicated job runner (Airflow, Argo, GitHub Actions).
  - Use a model registry (MLflow, Azure ML).
  - Push metrics to Prometheus/Grafana and use tracing (OpenTelemetry).
  - Replace fake LLM fallback with real Ollama endpoint, and implement rate limiting & cost controls.
- **Interview talking points**:
  - Explain why we treat confidence thresholds as routing rules.
  - Explain how you would add approval gates for retraining (validation set, shadow deployment, canary).
  - Discuss privacy/PII: avoid logging raw PII in production; use hashing/anonymization.
