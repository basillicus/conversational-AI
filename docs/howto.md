# How to run (quick)

1. Create virtualenv and install deps:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. Generate data and train:
   ```bash
   python prepare_data.py
   python train_model.py
   ```

3. Start app:
   ```bash
   uvicorn main:app --reload
   ```

4. Open Gradio UI:
   ```bash
   python gradio_app.py
   ```

5. Trigger retrain (demo):
   ```bash
   curl -X POST http://127.0.0.1:8000/retrain
   ```

6. View metrics:
   ```bash
   curl http://127.0.0.1:8000/metrics
   ```
