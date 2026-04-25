FROM python:3.12-slim

# Create a non-root user (Hugging Face Spaces requirement)
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

# Copy application files
COPY --chown=user . /app

# Install the dependencies from pyproject.toml
RUN pip install --no-cache-dir .

# Hugging Face Spaces listen on port 7860
EXPOSE 7860

# Start the OpenEnv FastAPI server
CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "7860"]
