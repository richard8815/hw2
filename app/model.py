import cv2
import mediapipe as mp
import numpy as np
from PIL import Image

from app.schemas import FacialFeatures, PersonalityTrait

mp_face_mesh = mp.solutions.face_mesh

# MediaPipe Face Mesh 랜드마크 인덱스 (468개 중 주요 포인트)
# 참고: https://github.com/google/mediapipe/blob/master/mediapipe/modules/face_geometry/data/canonical_face_model_uv_visualization.png
LANDMARKS = {
    "forehead_top": 10,
    "chin_bottom": 152,
    "left_face": 234,
    "right_face": 454,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "left_eye_outer": 33,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "nose_bridge": 6,
    "upper_lip_top": 13,
    "lower_lip_bottom": 14,
    "left_jaw": 172,
    "right_jaw": 397,
    "left_eyebrow_top": 70,
    "right_eyebrow_top": 300,
}


def _distance(p1: np.ndarray, p2: np.ndarray) -> float:
    return float(np.linalg.norm(p1 - p2))


def extract_features(image: Image.Image) -> FacialFeatures | None:
    """이미지에서 얼굴 랜드마크를 추출하고 관상학적 특징 비율을 계산한다."""
    img_array = np.array(image)
    if img_array.shape[2] == 4:  # RGBA -> RGB
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)

    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
    ) as face_mesh:
        results = face_mesh.process(img_array)

    if not results.multi_face_landmarks:
        return None

    landmarks = results.multi_face_landmarks[0].landmark
    h, w = img_array.shape[:2]

    def pt(name: str) -> np.ndarray:
        lm = landmarks[LANDMARKS[name]]
        return np.array([lm.x * w, lm.y * h])

    face_height = _distance(pt("forehead_top"), pt("chin_bottom"))
    face_width = _distance(pt("left_face"), pt("right_face"))
    eye_distance = _distance(pt("left_eye_inner"), pt("right_eye_inner"))
    nose_length = _distance(pt("nose_bridge"), pt("nose_tip"))
    lip_thickness = _distance(pt("upper_lip_top"), pt("lower_lip_bottom"))
    jaw_width = _distance(pt("left_jaw"), pt("right_jaw"))
    forehead_height = _distance(
        pt("forehead_top"),
        np.mean([pt("left_eyebrow_top"), pt("right_eyebrow_top")], axis=0),
    )

    return FacialFeatures(
        face_width_ratio=round(face_width / face_height, 3),
        eye_distance_ratio=round(eye_distance / face_width, 3),
        nose_length_ratio=round(nose_length / face_height, 3),
        lip_thickness_ratio=round(lip_thickness / face_height, 3),
        jaw_width_ratio=round(jaw_width / face_width, 3),
        forehead_ratio=round(forehead_height / face_height, 3),
    )


def predict_personality(features: FacialFeatures) -> list[PersonalityTrait]:
    """관상학 규칙 기반으로 성격 특성을 예측한다."""
    traits: list[PersonalityTrait] = []

    # 1) 얼굴 가로세로 비율 → 자신감 / 신중함
    if features.face_width_ratio > 0.75:
        traits.append(PersonalityTrait(
            trait="자신감",
            score=min(round(features.face_width_ratio * 100, 1), 95),
            description="넓은 얼굴형은 관상학에서 강한 자신감과 추진력을 나타냅니다.",
        ))
    else:
        traits.append(PersonalityTrait(
            trait="신중함",
            score=min(round((1 - features.face_width_ratio) * 100, 1), 95),
            description="갸름한 얼굴형은 섬세하고 신중한 성격을 나타냅니다.",
        ))

    # 2) 눈 사이 거리 → 포용력 / 집중력
    if features.eye_distance_ratio > 0.28:
        traits.append(PersonalityTrait(
            trait="포용력",
            score=min(round(features.eye_distance_ratio * 250, 1), 95),
            description="눈 사이 거리가 넓으면 타인을 잘 수용하고 열린 마음을 가집니다.",
        ))
    else:
        traits.append(PersonalityTrait(
            trait="집중력",
            score=min(round((0.4 - features.eye_distance_ratio) * 250, 1), 95),
            description="눈 사이가 가까우면 한 가지에 깊이 몰입하는 성향이 강합니다.",
        ))

    # 3) 코 길이 → 리더십
    if features.nose_length_ratio > 0.18:
        traits.append(PersonalityTrait(
            trait="리더십",
            score=min(round(features.nose_length_ratio * 400, 1), 95),
            description="코가 길면 자존심이 강하고 리더십을 발휘하는 성향입니다.",
        ))
    else:
        traits.append(PersonalityTrait(
            trait="친화력",
            score=min(round((0.25 - features.nose_length_ratio) * 400, 1), 95),
            description="코가 짧으면 사교적이고 친근한 인상을 줍니다.",
        ))

    # 4) 입술 두께 → 감성 / 이성
    if features.lip_thickness_ratio > 0.06:
        traits.append(PersonalityTrait(
            trait="감성적",
            score=min(round(features.lip_thickness_ratio * 1000, 1), 95),
            description="입술이 두꺼우면 감성이 풍부하고 표현력이 뛰어납니다.",
        ))
    else:
        traits.append(PersonalityTrait(
            trait="이성적",
            score=min(round((0.1 - features.lip_thickness_ratio) * 1000, 1), 95),
            description="입술이 얇으면 논리적이고 이성적인 판단을 선호합니다.",
        ))

    # 5) 턱 너비 → 끈기
    if features.jaw_width_ratio > 0.65:
        traits.append(PersonalityTrait(
            trait="끈기",
            score=min(round(features.jaw_width_ratio * 120, 1), 95),
            description="턱이 넓으면 인내심이 강하고 끈기 있는 성격입니다.",
        ))
    else:
        traits.append(PersonalityTrait(
            trait="유연함",
            score=min(round((1 - features.jaw_width_ratio) * 120, 1), 95),
            description="턱이 좁으면 적응력이 뛰어나고 유연한 사고를 합니다.",
        ))

    # 6) 이마 비율 → 지적 호기심
    if features.forehead_ratio > 0.2:
        traits.append(PersonalityTrait(
            trait="지적 호기심",
            score=min(round(features.forehead_ratio * 350, 1), 95),
            description="이마가 넓으면 지적 호기심이 강하고 분석적입니다.",
        ))
    else:
        traits.append(PersonalityTrait(
            trait="실행력",
            score=min(round((0.3 - features.forehead_ratio) * 350, 1), 95),
            description="이마가 좁으면 생각보다 행동이 앞서는 실행형입니다.",
        ))

    return traits


def generate_summary(traits: list[PersonalityTrait]) -> str:
    top_traits = sorted(traits, key=lambda t: t.score, reverse=True)[:3]
    names = ", ".join(t.trait for t in top_traits)
    return (
        f"당신의 관상에서 가장 두드러지는 성격 특성은 [{names}]입니다. "
        f"가장 강한 특성은 '{top_traits[0].trait}'({top_traits[0].score}점)이며, "
        f"{top_traits[0].description}"
    )
