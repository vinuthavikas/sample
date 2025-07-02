FROM python:3.10-slim

WORKDIR /app

# Copy only requirements.txt first (for caching)
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

CMD ["python", "app.py"]
