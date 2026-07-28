# Image AI Service

Barrier Free Friends MSA의 AI 사진 검증 서비스입니다. 사용자가 제보한 이미지를 Vision-Language 모델로 분석해 장애물 여부와 태그를 판별합니다.

전체 서비스 구성은 [조직 README](https://github.com/Barrier-Free-Friends)를 참고하세요.

## 기술 스택
- Python 3.11, FastAPI
- PyTorch, HuggingFace Transformers (VLM: `vikhyatk/moondream2`)
- aiokafka, py_eureka_client
- Prometheus FastAPI Instrumentator

## 동작 방식
- `report-service`가 발행한 이미지 검증 요청(`ai-verification-request` 토픽)을 Kafka로 구독
- 이미지를 다운로드해 VLM으로 추론, 장애물 여부/태그를 분류
- 결과를 `ai-verification-result` 토픽으로 발행
- 기동 시 Eureka에 등록되어 서비스 디스커버리에 참여

## 주요 구조
- `app/main.py` — 앱 진입점, lifespan에서 모델/Kafka/Eureka 초기화
- `app/api/routes.py` — `/analyze`, `/actuator/health`, `/info`
- `app/core/kafka_handler.py` — Kafka consumer/producer
- `app/services/ai_service.py` — 이미지 분석 로직

## 실행

```bash
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port ${INSTANCE_PORT:-8000}
```

## Docker

```bash
docker build -t image-ai-service .
docker run -p 8000:8000 image-ai-service
```
