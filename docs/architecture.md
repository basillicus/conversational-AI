# Architecture

The demo implements a compact, coherent system that showcases the AI/ML lifecycle.

```mermaid
flowchart LR
  A[User input / UI] --> B[FastAPI /chat endpoint]
  B --> C{Routing logic}
  C -->|Local & confident| D[Local Intent Model (TF-IDF + LR)]
  C -->|LLM chosen| E[Ollama LLM or fallback]
  D --> F[Telemetry logger (logs/chat_metrics.log)]
  E --> F
  F --> G[Monitoring scripts (/metrics, monitoring.py)]
  H[Feedback collection (/submit_feedback)] --> I[Retraining pipeline (train_model.py)]
  I --> D
```

This diagram shows the lifecycle: Data -> Train -> Deploy -> Monitor -> Retrain.
