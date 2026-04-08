from pydantic import BaseModel


class FacialFeatures(BaseModel):
    face_width_ratio: float       # 얼굴 가로세로 비율
    eye_distance_ratio: float     # 눈 사이 거리 비율
    nose_length_ratio: float      # 코 길이 비율
    lip_thickness_ratio: float    # 입술 두께 비율
    jaw_width_ratio: float        # 턱 너비 비율
    forehead_ratio: float         # 이마 비율


class PersonalityTrait(BaseModel):
    trait: str
    score: float
    description: str


class PredictionResponse(BaseModel):
    success: bool
    face_detected: bool
    features: FacialFeatures | None = None
    personalities: list[PersonalityTrait] = []
    summary: str = ""


class HealthResponse(BaseModel):
    status: str
    version: str
