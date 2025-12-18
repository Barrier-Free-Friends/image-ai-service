# 베이스 이미지 설정
FROM python:3.11

# 버퍼링 말고 즉시 출력 설정
ENV PYTHONUNBUFFERED=1

# 작업 디렉토리 설정
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "$INSTANCE_PORT"]