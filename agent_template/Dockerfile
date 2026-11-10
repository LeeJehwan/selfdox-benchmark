# Template for Phase 3 agents. Reads /input/*.jpg, writes /output/{predictions.json,images/}.
# The container runs without network access (see docker-compose.yml).
FROM python:3.12-slim
WORKDIR /agent
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY run.py .
ENTRYPOINT ["python", "run.py", "--input", "/input", "--output", "/output"]
