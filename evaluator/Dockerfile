# Official SelfDox scorer: CPU only, no ML model, pinned by uv.lock.
FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.7.13 /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY selfdox ./selfdox
RUN uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"
ENTRYPOINT ["python", "-m", "selfdox.score"]
