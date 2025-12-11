from fastapi import FastAPI, HTTPException
from pydantic import BaseModel # json 바디 파싱용
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import io
import requests # URL에서 이미지 다운로드하는 도구

app = FastAPI()

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

class ImageRequest(BaseModel):
    imageUrl : str
    

@app.post("/analyze")
async def analyze_image(request: ImageRequest):
    
    
    try:
        
        # 요청 거부하지 않도록 User-Agent 헤더 추가
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
        }
        
        # 1. 이미지 다운로드
        response = requests.get(request.imageUrl, headers=headers)
        response.raise_for_status()
        
        content_type = response.headers.get('Content-Type', '')
        print(f"다운로드한 이미지의 Content-Type: {content_type}")
        
        if 'image' not in content_type:
             raise HTTPException(status_code=400, detail=f"URL이 이미지가 아닙니다. 감지된 타입: {content_type}")

        # [수정 포인트 3] 안전하게 이미지 열기
        image = Image.open(io.BytesIO(response.content))
        
    except requests.exceptions.RequestException as e:
        print(f"다운로드 에러: {e}")
        raise HTTPException(status_code=400, detail=f"이미지 다운로드 실패(네트워크): {str(e)}")
    except Exception as e:
        print(f"이미지 처리 에러: {e}")
        raise HTTPException(status_code=400, detail=f"이미지 변환 실패: {str(e)}")

    # 3. 질문 (프롬프트) 변경
    prompt = "Check if there is a physical object blocking the path, such as a fallen tree, construction barrier, or large rocks etc. Do not consider snow, leaves, or a hill as an obstacle unless it completely blocks the way. If the path is passable for a wheelchair, say 'The path is clear'. Otherwise, describe the blocking object."

    
    # 4. 모델 추론
    enc_image = model.encode_image(image)
    answer = model.answer_question(enc_image, prompt, tokenizer)

    print(f"AI 답변: {answer}") # 로그 확인용

    # 5. 결과 후처리
    is_obstacle = False
    tag = "normal"
    answer_lower = answer.lower()

    # 모델이 "The path is clear"라고 답하면 장애물 없음으로 처리
    # 부정적인 단어(clear, passable)가 포함되어 있으면 장애물 없음
    if "clear" in answer_lower or "passable" in answer_lower or "no obstacle" in answer_lower or 'no' in answer_lower:
        is_obstacle = False
        tag = "normal"
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
            tag = "construction_site"
        elif any(k in answer_lower for k in tree_keywords):
            tag = "fallen_tree"
        elif any(k in answer_lower for k in rock_keywords):
            tag = "rock"
        elif any(k in answer_lower for k in furniture_keywords):
            tag = "street_furniture"
        elif any(k in answer_lower for k in slope_keywords):
            tag = "steep_slope" # 계단 등
        else:
            # 눈(snow)이나 언덕(hill)이라서 장애물로 잡혔는데, 위 키워드에 없으면 
            # 사실 장애물이 아닐 확률이 높으므로 다시 한번 필터링하거나 기타로 분류
            if "snow" in answer_lower or "hill" in answer_lower:
                 # AI가 "Just snow"라고 대답했을 경우를 대비해 예외 처리 가능
                 # 하지만 위 prompt에서 clear를 유도했으므로 여기서는 기타 장애물로 둠
                 tag = "other_obstacle" 
            else:
                 tag = "other_obstacle"
            
    print(answer)
    # 6. JSON 응답
    return {
        "analysis_result": answer,
        "is_obstacle": is_obstacle,
        "tag": tag
    }