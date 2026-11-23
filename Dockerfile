# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.13-slim AS builder

# Prevent Python from writing .pyc files and enable unbuffered console logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

RUN pip install uv

# Create a virtual environment and install dependencies directly
RUN uv venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# # Install required packages for the RAG System
#  RUN uv pip install streamlit chromadb ollama openai pypdf rank_bm25 python-dotenv langchain-text-splitters

# Copy the requirements file first to leverage Docker cache
COPY requirements.txt .
# Install base dependencies from the requirements file
RUN uv pip install -r requirements.txt

# Install lightweight OCR Python dependencies in a separate layer
# (limits concurrent uv downloads so slow networks / Docker DNS don't time out)
ENV UV_HTTP_TIMEOUT=300
ENV UV_CONCURRENT_DOWNLOADS=2
ENV UV_LINK_MODE=copy
COPY requirements-ocr.txt .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r requirements-ocr.txt

# ==========================================
# Stage 2: Runner (Production)
# ==========================================
FROM python:3.13-slim

# Prevent Python from writing .pyc files and enable unbuffered console logging
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install Tesseract OCR with Arabic & English language packs (for scanned PDFs) and curl (for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    tesseract-ocr-ara \
    tesseract-ocr-eng \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install poppler-utils (pdftoppm) for lightweight PDF-to-image rendering (supports JBIG2)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Activate the virtual environment from the builder stage
ENV PATH="/opt/venv/bin:$PATH"

# Create a non-root user for security purposes
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY . .

# Force full ownership of the entire /app directory and its contents to appuser
RUN chown -R appuser:appuser /app

# Switch to the non-root user
USER appuser

# Expose Streamlit port
EXPOSE 8501

# Run the Streamlit app
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]