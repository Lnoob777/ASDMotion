import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# -------- MODEL PATHS --------
FACE_MODEL = "assets/video/face_landmarker.task"
POSE_MODEL = "assets/video/pose_landmarker_full.task"

# -------- FACE LANDMARKER --------
face_options = vision.FaceLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=FACE_MODEL),
    running_mode=vision.RunningMode.IMAGE
)
face_landmarker = vision.FaceLandmarker.create_from_options(face_options)

# -------- POSE LANDMARKER --------
pose_options = vision.PoseLandmarkerOptions(
    base_options=python.BaseOptions(model_asset_path=POSE_MODEL),
    running_mode=vision.RunningMode.IMAGE
)
pose_landmarker = vision.PoseLandmarker.create_from_options(pose_options)


def extract_landmarks(frame):
    """
    Input:
        frame: BGR image (OpenCV)

    Output:
        face_landmarks: list of landmarks OR None
        pose_landmarks: list of 33 landmarks OR None
    """

    rgb = frame[:, :, ::-1]

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    face_result = face_landmarker.detect(mp_image)
    pose_result = pose_landmarker.detect(mp_image)

    # ---- FACE (list[list[landmarks]]) ----
    face_landmarks = (
        face_result.face_landmarks[0]
        if face_result.face_landmarks and len(face_result.face_landmarks) > 0
        else None
    )

    # ---- POSE (list[list[landmarks]]) ----
    pose_landmarks = (
        pose_result.pose_landmarks[0]
        if pose_result.pose_landmarks and len(pose_result.pose_landmarks) > 0
        else None
    )

    return face_landmarks, pose_landmarks
