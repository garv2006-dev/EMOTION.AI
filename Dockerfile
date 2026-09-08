FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY . .

# Expose Flask web server port
EXPOSE 5000

# Set Environment variables
ENV PORT=5000
ENV PYTHONUNBUFFERED=1

# Run Flask application using Gunicorn WSGI server
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 2 app:app"]
