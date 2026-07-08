FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Hugging Face Spaces requires the container to listen on port 7860
# specifically -- it isn't configurable to a different port.
ENV PORT=7860
EXPOSE 7860

CMD ["python", "api/app.py"]
