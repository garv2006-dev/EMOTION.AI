FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Run Gradio application
CMD ["python", "app.py"]
