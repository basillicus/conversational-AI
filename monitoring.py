"""monitoring.py

Simple helper to parse logs/chat_metrics.log and generate a small report.
Intended to be educational: in production you would stream metrics to Prometheus/Grafana
and use observability tools.

Run:
    python monitoring.py

Output:
    - prints aggregated stats
    - writes a small CSV of recent metrics for quick inspection
"""
import json
from pathlib import Path
from collections import Counter

LOG = Path("logs/chat_metrics.log")
OUT = Path("logs/metrics_summary.csv")

if not LOG.exists():
    print("No logs found. Interact with the app to generate chat logs.")
    raise SystemExit(0)

lines = [l.strip() for l in LOG.read_text(encoding="utf-8").splitlines() if l.strip()]
objs = [json.loads(l) for l in lines]
count = len(objs)
avg_latency = sum(o.get("latency_ms",0) for o in objs)/count if count else 0
anomalies = [o for o in objs if o.get("anomaly")]
intents = Counter(o.get("intent","unknown") for o in objs)

print(f"Total requests: {count}")
print(f"Average latency (ms): {avg_latency:.2f}")
print(f"Anomaly count: {len(anomalies)}")
print("Top intents:")
for k,v in intents.most_common(10):
    print(f"  {k}: {v}")

# Write a CSV summary with timestamp, latency, intent
with OUT.open("w", encoding="utf-8") as f:
    f.write("ts,latency_ms,intent,anomaly\n")
    for o in objs[-200:]:
        f.write(f"{o.get('ts')},{o.get('latency_ms')},{o.get('intent')},{o.get('anomaly')}\n")
print(f"Wrote summary to {OUT}")
