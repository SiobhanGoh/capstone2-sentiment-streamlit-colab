# -----------------------------
# Base Image
# -----------------------------
FROM python:3.10-slim

# -----------------------------
# System Packages
# -----------------------------
RUN apt-get update && apt-get install -y \
    git \
    libglib2.0-0 \
    libsm6 \
    libxrender1 \
    libxext6 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# -----------------------------
# Working Directory
# -----------------------------
WORKDIR /app

# -----------------------------
# Install Python Dependencies
# -----------------------------
COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# -----------------------------
# Copy Application Files
# -----------------------------
COPY src/ ./src/

# -----------------------------
# Expose Streamlit Port
# -----------------------------
EXPOSE 8501

# -----------------------------
# Streamlit Environment Fix
# (Required for Hugging Face Spaces)
# -----------------------------
ENV PORT=7860
ENV STREAMLIT_SERVER_PORT=7860
ENV STREAMLIT_SERVER_HEADLESS=true

# Disable CORS + XSRF to avoid HF iframe issues
ENV STREAMLIT_SERVER_ENABLE_CORS=false
ENV STREAMLIT_SERVER_ENABLE_XSRF_PROTECTION=false

# -----------------------------
# Launch Streamlit App
# -----------------------------
CMD ["streamlit", "run", "src/streamlit_app.py", "--server.port=7860", "--server.address=0.0.0.0"]
