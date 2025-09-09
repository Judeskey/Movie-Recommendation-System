FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \        pip install --no-cache-dir -r requirements.txt

COPY offline/ offline/
COPY service/ service/
COPY tools/ tools/
COPY data/ data/
COPY models/ models/

EXPOSE 8000
CMD ["python", "service/server.py"]
