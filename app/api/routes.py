from fastapi import HTTPException, APIRouter
from app.core.config import APP_NAME
from app.schemas.image_dto import ImageListRequest, AnalysisResult
from app.services.ai_service import ai_service
# from concurrent.futures import ProcessPoolExecutor
import asyncio
from app.main import thread_executor


router = APIRouter()

@router.get("/actuator/health")
async def health_check():
    return {"status": "UP"}

@router.get("/")
async def root():
    return {"message": "Image AI Service is running."}

@router.get("/info")
async def info():
    return {"app": APP_NAME, "status": "running"}


@router.post("/analyze", response_model=AnalysisResult)
async def analyze_list_image(request: ImageListRequest):
    
    """
    여러 이미지 URL을 받아 for 루프를 돌려 분석한다,
    하나라도 '장애물이 아님' (is_obstacle = False)으로 판단되면 결과를 리턴한다
    모든 이미지가 장애물일 경우, 첫 번째 장애물 결과를 리턴하도록 한다
    """
    
    if not request.images:
        raise HTTPException(status_code=400, detail="이미지 URL 리스트가 비어 있습니다.")
    
    first_obstacle_result = None
    
    # 현재 이벤트 루프
    loop = asyncio.get_running_loop()
    
    for url in request.images:
        # 단일 이미지 분석하고 결과 받기
        # result = await ai_service.analyze_single_image(url)
        
        # 스레드 풀에서 동기 함수를 비동기적으로 실행
        result = await loop.run_in_executor(thread_executor, ai_service.analyze_single_image, url)
        
        print(f"이미지 URL: {url.fileUrl}, 분석 결과: {result}")
        
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
