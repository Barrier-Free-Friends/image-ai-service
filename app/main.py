from fastapi import FastAPI
import uvicorn
from contextlib import asynccontextmanager
import socket
import py_eureka_client.eureka_client as eureka_client
from app.core.config import APP_NAME, INSTANCE_PORT, EUREKA_SERVER_URL, get_external_ip
from app.services.ai_service import ai_service
from app.api.routes import router

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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=INSTANCE_PORT)