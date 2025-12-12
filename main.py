from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # json 바디 파싱용
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
import requests # URL에서 이미지 다운로드하는 도구
import uvicorn
from contextlib import asynccontextmanager
import socket
import py_eureka_client.eureka_client as eureka_client
from typing import List

APP_NAME = "IMAGE-AI-SERVICE"
INSTANCE_PORT = 8000
EUREKA_SERVER_URL = "http://52.79.151.83:3150/eureka/"

try:
    hostname = socket.gethostname()
    INSTANCE_IP = socket.gethostbyname(hostname)
except:
    INSTANCE_IP = "127.0.0.1"


def get_external_ip():
    try:
        ip = requests.get('https://api.ipify.org').text
        print(f"퍼블릭 IP 주소: {ip}")
        return ip
    except Exception as e:
        print(f"퍼블릭 IP 주소 조회 실패: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # 앱 시작 시 실행할 코드
    
    cur_ip = get_external_ip()
    
    await eureka_client.init_async(
        eureka_server=EUREKA_SERVER_URL,
        app_name=APP_NAME,
        instance_port=INSTANCE_PORT,
        instance_ip=cur_ip,
        instance_host=cur_ip
    )
    print("Eureka 등록 완료")
    yield
    await eureka_client.stop()
    print("Eureka 등록 해제 완료")


app = FastAPI(lifespan=lifespan)
# 1. 모델 설정
model_id = "vikhyatk/moondream2"
revision = "2024-08-26"

print("모델 로딩 중... 잠시만 기다려주세요.")

#1: device_map="cpu" 삭제!
model = AutoModelForCausalLM.from_pretrained(
    model_id, 
    trust_remote_code=True, 
    revision=revision,
    attn_implementation="eager"
)

#2: 모델을 명시적으로 CPU로 이동
model.to("cpu")
model.eval() # 평가 모드로 설정 (속도 및 안정성 향상)

tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
print("모델 로딩 완료")

    
@app.get("/actuator/health")
async def health_check():
    return {"status": "UP"}

@app.get("/")
async def root():
    return {"message": "Image AI Service is running."}

@app.get("/info")
async def info():
    return {"app": APP_NAME, "status": "running"}

class ImageListRequest(BaseModel):
    imageUrls : List[str]
        
class AnalysisResult(BaseModel):
    analysis_result: str
    is_obstacle: bool
    tag: str

class ImageReqeustDto(BaseModel):
    fileUrl : str
    latitude : float
    longitude : float
    address : str

def analyze_single_image(image_request: ImageReqeustDto) -> AnalysisResult:
    try:
        
        # 요청 거부하지 않도록 User-Agent 헤더 추가
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        
        # 1. 이미지 다운로드
        response = requests.get(image_request.fileUrl, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        print(f"다운로드한 이미지의 Content-Type: {content_type}")
        
        if 'image' not in content_type:
             raise HTTPException(status_code=400, detail=f"URL이 이미지가 아닙니다. 감지된 타입: {content_type}")

        # [수정 포인트 3] 안전하게 이미지 열기
        image = Image.open(io.BytesIO(response.content))
        
    except requests.exceptions.RequestException as e:
        print(f"다운로드 에러: {e}")
        return AnalysisResult(analysis_result=f"이미지 다운로드 실패(네트워크): {str(e)}", is_obstacle=False, tag="download error")
    except Exception as e:
        print(f"이미지 처리 에러: {e}")
        raise AnalysisResult(analysis_result=f"이미지 변환 실패: {str(e)}", is_obstacle=False, tag="process error")

    # 3. 질문 (프롬프트) 변경
    prompt = (
            "Is there a 'fallen tree', 'construction barrier', 'large rocks', 'stairs', or 'furniture' blocking the way? "
            "If yes, name the obstacle. "
            "If no obstacle is seen, check if this is a road or path. "
            "If it is NOT a road or path, say 'not a path'. "
            "If it is a clear road, say 'clear'."
        )
    
    # 4. 모델 추론
    enc_image = model.encode_image(image)
    answer = model.answer_question(enc_image, prompt, tokenizer)

    print(f"AI 답변: {answer}") # 로그 확인용

    # 5. 결과 후처리
    is_obstacle = False
    tag = "normal"
    answer_lower = answer.lower()
    
    if "not a path" in answer_lower:
        # 길이 아닌 경우
        is_obstacle = False
        tag = "not_a_path"
        answer = "No"

    # 길 깨끗한 경우
    elif "clear" in answer_lower or "passable" in answer_lower or "no obstacle" in answer_lower or 'no' in answer_lower:
        is_obstacle = False
        tag = "normal"
        answer = "No"
    
    # 장애물 있는 경우
    else:
        # 그 외의 경우 (무언가 설명하기 시작함) -> 장애물로 간주
        is_obstacle = True
        
        tree_keywords = ["tree", "branch", "log", "trunk", "wood", "root", "plant", "bush", "stump"]
        rock_keywords = ["rock", "stone", "boulder", "concrete", "rubble", "brick"]
        construction_keywords = ["construction", "cone", "barrier", "sign", "work", "safety", "fence"]
        furniture_keywords = ["planter", "pot", "box", "bench", "pole", "bollard", "post", "street furniture"]
        # stairs나 step은 확실한 장애물이므로 유지하되, slope(단순 경사)는 제외할지 고민 필요
        slope_keywords = ["stairs", "step", "staircase"] 

        if any(k in answer_lower for k in construction_keywords):
            tag = "construction"
        elif any(k in answer_lower for k in tree_keywords):
            tag = "tree"
        elif any(k in answer_lower for k in rock_keywords):
            tag = "rock"
        elif any(k in answer_lower for k in furniture_keywords):
            tag = "furniture"
        elif any(k in answer_lower for k in slope_keywords):
            tag = "slope" # 계단 등
        else:
            # 눈(snow)이나 언덕(hill)이라서 장애물로 잡혔는데, 위 키워드에 없으면 
            # 사실 장애물이 아닐 확률이 높으므로 다시 한번 필터링하거나 기타로 분류
            if "snow" in answer_lower or "hill" in answer_lower:
                 # AI가 "Just snow"라고 대답했을 경우를 대비해 예외 처리 가능
                 # 하지만 위 prompt에서 clear를 유도했으므로 여기서는 기타 장애물로 둠
                 tag = "other_obstacle" 
            else:
                 tag = "other_obstacle"
        answer = "Yes"
        return AnalysisResult(analysis_result=answer, is_obstacle=is_obstacle, tag=tag)
    
    # 장애물이 아닐 경우            
    print(answer)
    return AnalysisResult(analysis_result=answer, is_obstacle=is_obstacle, tag=tag)


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_list_image(request: ImageListRequest):
    
    """
    여러 이미지 URL을 받아 for 루프를 돌려 분석한다,
    하나라도 '장애물이 아님' (is_obstacle = False)으로 판단되면 결과를 리턴한다
    모든 이미지가 장애물일 경우, 첫 번째 장애물 결과를 리턴하도록 한다
    """
    
    if not request.imageUrls:
        raise HTTPException(status_code=400, detail="이미지 URL 리스트가 비어 있습니다.")
    
    first_obstacle_result = None
    
    for url in request.imageUrls:
        # 단일 이미지 분석하고 결과 받기
        result = analyze_single_image(url)
        
        # 장애물이 아닌 경우
        if not result.is_obstacle:
            return result.model_dump()
        
        # 장애물인 경우, 첫 번째 결과 저장
        if first_obstacle_result is None:
            first_obstacle_result = result.model_dump()

    # 모든 이미지가 장애물인 경우, 첫 번째 장애물 결과 리턴
    if first_obstacle_result is not None:
        return first_obstacle_result

    return AnalysisResult(
        analysis_result="No",
        is_obstacle=False,
        tag="normal"
    ).model_dump()
    
    



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=INSTANCE_PORT)