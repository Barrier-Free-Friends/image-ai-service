from pydantic import BaseModel
from typing import List

class AnalysisResult(BaseModel):
    analysis_result: str
    is_obstacle: bool
    tag: str

class AiImageRequest(BaseModel):
    fileUrl : str
    latitude : float
    longitude : float
    address : str

class ImageListRequest(BaseModel):
    images : List[AiImageRequest]
