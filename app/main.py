from fastapi import FastAPI
from fastapi.openapi.utils import get_openapi
import uvicorn
from contextlib import asynccontextmanager
import socket
import py_eureka_client.eureka_client as eureka_client
from app.core.config import APP_NAME, INSTANCE_PORT, EUREKA_SERVER_URL, get_external_ip
from app.services.ai_service import ai_service
from app.api.routes import router
from prometheus_fastapi_instrumentator import Instrumentator

try:
    hostname = socket.gethostname()
    INSTANCE_IP = socket.gethostbyname(hostname)
except:
    INSTANCE_IP = "127.0.0.1"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행할 코드
    

    
    cur_ip = get_external_ip()
    ai_service.load_model()
    
    await eureka_client.init_async(
        eureka_server=EUREKA_SERVER_URL,
        app_name=APP_NAME,
        instance_port=INSTANCE_PORT,
        instance_ip=cur_ip,
        instance_host=cur_ip
    )
    print("Eureka 등록 완료")
    yield # 앱 실행 중
    
    
    await eureka_client.stop()
    print("Eureka 등록 해제 완료")

app = FastAPI(lifespan=lifespan)
app.include_router(router)

def custom_openapi():
    openapi_schema = get_openapi(
        title=f"{APP_NAME} API",
        version="1.0.0",
        description="Image AI Service API 문서",
        routes=app.routes,
    )
    
    openapi_schema["openapi"] = "3.0.0"
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema

app.openapi = custom_openapi

instrumentator = Instrumentator(
    should_group_status_codes=False,
    should_ignore_untemplated=True,
    should_instrument_requests_inprogress=True,
    
)

instrumentator.instrument(app).expose(app, endpoint="/actuator/prometheus")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=INSTANCE_PORT)