def compute_quality_mask(
    frame_id,
    face_landmarks,
    pose_landmarks
):
    """
    Returns a quality dictionary for one frame
    """

    pose_valid = 1 if pose_landmarks is not None else 0
    face_valid = 1 if face_landmarks is not None else 0

    # Pose is mandatory, face is optional
    frame_valid = 1 if pose_valid == 1 else 0

    return {
        "frame_id": frame_id,
        "pose_valid": pose_valid,
        "face_valid": face_valid,
        "frame_valid": frame_valid
    }
