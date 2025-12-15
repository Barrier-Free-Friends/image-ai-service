import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from app.schemas.image_dto import AiImageRequest, AnalysisResult
from fastapi import HTTPException
import requests
from PIL import Image
import io

class AiService:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        # 모델 설정
        self.model_id = "vikhyatk/moondream2"
        self.revision = "2024-08-26"
    
    def load_model(self):
        print("모델 로딩 중... 잠시만 기다려주세요.")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_id, 
            trust_remote_code=True, 
            revision=self.revision,
            attn_implementation="eager"
        )
        self.model.to("cpu")
        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, revision=self.revision)
        print("모델 로딩 완료")


    def analyze_single_image(self, image_request: AiImageRequest) -> AnalysisResult:
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

            # 안전하게 이미지 열기
            image = Image.open(io.BytesIO(response.content))
            
        except requests.exceptions.RequestException as e:
            print(f"다운로드 에러: {e}")
            return AnalysisResult(analysis_result=f"이미지 다운로드 실패(네트워크): {str(e)}", is_obstacle=False, tag="download error")
        except Exception as e:
            print(f"이미지 처리 에러: {e}")
            raise AnalysisResult(analysis_result=f"이미지 변환 실패: {str(e)}", is_obstacle=False, tag="process error")

        # 3. 질문
        prompt = (
                "Is there a 'fallen tree', 'construction barrier', 'large rocks', 'stairs', or 'furniture' blocking the way? "
                "If yes, name the obstacle. "
                "If no obstacle is seen, check if this is a road or path. "
                "If it is NOT a road or path, say 'not a path'. "
                "If it is a clear road, say 'clear'."
            )
        
        # 4. 모델 추론
        enc_image = self.model.encode_image(image)
        answer = self.model.answer_question(enc_image, prompt, self.tokenizer)

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
        return AnalysisResult(analysis_result=answer, is_obstacle=is_obstacle, tag=tag)\

ai_service = AiService()