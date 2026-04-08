import uuid
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from app.model import extract_features, generate_summary, predict_personality
from app.schemas import HealthResponse, PredictionResponse

UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}
MAX_SIZE_MB = 10

app = FastAPI(
    title="Face Personality API",
    description="얼굴 이미지를 업로드하면 관상학 기반으로 성격을 예측합니다.",
    version="0.1.0",
)


@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(..., description="얼굴 이미지 (JPG/PNG/WebP, 최대 10MB)")):
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(400, f"지원하지 않는 파일 형식입니다: {file.content_type}")

    contents = await file.read()
    if len(contents) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(400, f"파일 크기가 {MAX_SIZE_MB}MB를 초과합니다.")

    # 임시 저장
    ext = file.filename.rsplit(".", 1)[-1] if file.filename else "jpg"
    save_path = UPLOAD_DIR / f"{uuid.uuid4().hex}.{ext}"
    save_path.write_bytes(contents)

    try:
        image = Image.open(save_path).convert("RGB")
        features = extract_features(image)
    except Exception as e:
        raise HTTPException(422, f"이미지 처리 중 오류가 발생했습니다: {e}")
    finally:
        save_path.unlink(missing_ok=True)

    if features is None:
        return PredictionResponse(
            success=True,
            face_detected=False,
            summary="얼굴을 감지하지 못했습니다. 정면 얼굴이 잘 보이는 사진을 올려주세요.",
        )

    traits = predict_personality(features)
    summary = generate_summary(traits)

    return PredictionResponse(
        success=True,
        face_detected=True,
        features=features,
        personalities=traits,
        summary=summary,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
