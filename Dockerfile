FROM python:3.11-slim
WORKDIR /app

# Install dependencies first (layer caching)
COPY api/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api/app.py ./app.py

# Copy model artefact (produced by training stage)
COPY model/price_model.pkl ./model/price_model.pkl

# Non-root user for security
RUN useradd -m appuser && chown -R appuser /app
USER appuser

ENV MODEL_PATH=model/price_model.pkl
ENV PYTHONUNBUFFERED=1
EXPOSE 5000

# 2 workers per CPU core is standard for CPU-bound ML inference
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5000", "--timeout", "120", "api.app:app"]
