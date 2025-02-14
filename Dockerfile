# Use Python 3.12 slim variant
FROM python:3.12-slim-bookworm

# Install required system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates nodejs npm sqlite3 tesseract-ocr poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Install `uv`
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure `uv` is available in PATH
ENV PATH="/root/.local/bin:$PATH"

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies **system-wide**
RUN uv pip install --system \
    fastapi uvicorn requests python-dotenv \
    pillow pytesseract openai tiktoken numpy pandas \
    sentence-transformers sqlite-utils

# Ensure `npx` works by installing Prettier globally
RUN npm install -g prettier@3.4.2

# Expose the application port
EXPOSE 8000

# Start the FastAPI app
CMD ["uv", "run", "app.py"]
