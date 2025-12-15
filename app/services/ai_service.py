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
                """Classify the image into exactly ONE tag based on walkability.
                
                If the image does NOT show a walking path (e.g., animals, people, portraits,
                indoor scenes, backgrounds, illustrations) AND no obstacle is blocking a path,
                output word is clear

                If a person can walk normally and no obstacle is blocking the way,
                output word is clear

                If an obstacle exists, output ONE of the following tags:
                construction
                tree
                rock
                furniture
                stairs
                not_a_path
                other_obstacle

                Output ONLY the tag word. Do NOT answer yes or no."""
                            )
        
        # 4. 모델 추론
        enc_image = self.model.encode_image(image)
        raw_answer = self.model.answer_question(enc_image, prompt, self.tokenizer)
        
        clean_answer = raw_answer.strip().lower()
        clean_answer = clean_answer.replace(".", "").replace("'", "").replace('"', "")

        print(f"AI 답변: {raw_answer}") # 로그 확인용
        print(f"정제된 답변: {clean_answer}") # 로그 확인용

        # 5. 결과 후처리
        is_obstacle = True
        tag = "other_obstacle"
        analysis_result = "Yes"
        
        if 'clear' in clean_answer:
            is_obstacle = False
            tag = "normal"
            analysis_result = "No"
            
           
        elif "not a path" in clean_answer:
            is_obstacle = False
            tag = "not_a_path"
            analysis_result = "No"
         # 장애물 있는 경우
        elif "construction" in clean_answer:
            tag = "construction"
        elif "tree" in clean_answer:
            tag = "tree"
        elif "rock" in clean_answer:
            tag = "rock"
        elif "furniture" in clean_answer:
            tag = "furniture"
        elif "stairs" in clean_answer:
            tag = "slope"
        else:
            tag = "other_obstacle"
            
        return AnalysisResult(analysis_result=analysis_result, is_obstacle=is_obstacle, tag=tag)
        
        
ai_service = AiService()