import json
import asyncio
from app.services.ai_service import ai_service
from datetime import datetime
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
import uuid

class AiKafkaHandler:
    def __init__(self):
        self.bootstrap_servers = "34.64.55.76:9092,34.64.55.76:9093,34.64.55.76:9094"
        self.request_topic = "ai-verification-request"
        self.result_topic = "ai-verification-result"
        self.consumer = None
        self.producer = None

    async def start(self):
        self.consumer = AIOKafkaConsumer(
            self.request_topic,
            bootstrap_servers=self.bootstrap_servers,
            group_id="ai-service-group",
            value_deserializer=lambda v: json.loads(v.decode('utf-8')),
            max_poll_records=1,
            max_poll_interval_ms=300000,
            session_timeout_ms=30000,
            heartbeat_interval_ms=10000
        )
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers,
            value_serializer=lambda v: json.dumps(v).encode('utf-8')
        )
        await self.consumer.start()
        await self.producer.start()
        asyncio.create_task(self.consume_requests())

    async def consume_requests(self):
        async for msg in self.consumer:
            try:

                event_data = msg.value
                source_id = event_data.get("sourceId")
                req_images = event_data.get("images", [])
                
                print(f"Received Analysis Request: eventId={event_data.get('eventId')}")

                # AI 분석 실행
                analysis_results = []
                for img_data in req_images:
                    from app.schemas.image_dto import AiImageRequest
                    req = AiImageRequest(**img_data)
                    
                    # AI 서비스 호출
                    res = ai_service.analyze_single_image(req)
                    analysis_results.append(res.model_dump())

                # 결과 이벤트 생성 -> Spring의 AbstractDomainEvent 구조에 맞춤
                #	"__TypeId__": "org.bf.global.infrastructure.event.ReportCreatedEvent" 
                
                result_event = {
                    "eventId": str(uuid.uuid4()),
                    "occurredAt": datetime.now().isoformat(),
                    "sourceService": "image-ai-service",
                    # "results": analysis_results,
                    "analysis_result" : res.analysis_result,
                    "is_obstacle" : res.is_obstacle,
                    "tag" : res.tag,
                    "sourceId": source_id,
                    "sourceTable": ""
                }
                
                headers = [
                    ("__TypeId__", b"org.bf.reportservice.infrastructure.event.AiVerificationCompletedEvent")
                ]

                await self.producer.send_and_wait(
                    self.result_topic,
                    value=result_event,
                    key=result_event.get("eventId").encode('utf-8'),
                    headers=headers)
                print(f"Sent Analysis Result: eventId={result_event.get('eventId')}")
                
            except Exception as e:
                print(f"Error!!!!!!!!: {e}")

    async def stop(self):
        if self.consumer:
            await self.consumer.stop()
        if self.producer:
            await self.producer.stop()

kafka_handler = AiKafkaHandler()