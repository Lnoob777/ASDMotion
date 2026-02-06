import csv
import os
import cv2
import json

from src.pipeline.router import route_video
from src.models.video.mediapipe_layer.extractor import extract_landmarks
from src.models.video.mediapipe_layer.render_pose import render_pose
from src.models.video.mediapipe_layer.aligner import aligned_face_crop
from src.models.video.mediapipe_layer.quality import compute_quality_mask


# ================= CONFIG =================
CSV_PATH = r"C:\Users\yuvad\Desktop\csv\videos.csv"
TMIN = 2.0

SKELETON_OUT_ROOT = "data/processed/skeletons"
FACE_OUT_ROOT = "data/processed/faces"
QUALITY_OUT_ROOT = "data/processed/quality"
# ==========================================


def load_video(video_path):
    """
    Load a video and return frames, fps, duration
    """
    cap = cv2.VideoCapture(video_path)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0

    frames = []
    frame_id = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames.append({
            "frame_id": frame_id,
            "image": frame
        })
        frame_id += 1

    cap.release()
    return frames, fps, duration


def main():
    with open(CSV_PATH, newline="") as f:
        reader = csv.DictReader(f)

        # ================= VIDEO LOOP =================
        for row in reader:
            video_path = row["video_path"]

            # Extract case ID (case1, case2, ...)
            video_id = os.path.basename(
                os.path.dirname(video_path)
            )

            print(f"\nProcessing video: {video_path}")

            frames, fps, duration = load_video(video_path)
            route = route_video(duration, TMIN)

            print(f"FPS: {fps}")
            print(f"Total frames: {len(frames)}")
            print(f"Duration: {duration:.2f}s")
            print(f"Route: {route}")

            # Output directories
            skeleton_dir = os.path.join(SKELETON_OUT_ROOT, video_id)
            face_dir = os.path.join(FACE_OUT_ROOT, video_id)
            quality_dir = os.path.join(QUALITY_OUT_ROOT, video_id)

            os.makedirs(skeleton_dir, exist_ok=True)
            os.makedirs(face_dir, exist_ok=True)
            os.makedirs(quality_dir, exist_ok=True)

            quality_list = []

            skeleton_saved = 0
            face_saved = 0

            # ================= FRAME LOOP =================
            for item in frames:
                frame_id = item["frame_id"]
                frame = item["image"]

                # MediaPipe Tasks landmark extraction
                face_landmarks, pose_landmarks = extract_landmarks(frame)

                # ---------- QUALITY MASK ----------
                quality = compute_quality_mask(
                    frame_id=frame_id,
                    face_landmarks=face_landmarks,
                    pose_landmarks=pose_landmarks
                )
                quality_list.append(quality)

                # ---------- POSE → SKELETON ----------
                if pose_landmarks is not None:
                    skeleton_img = render_pose(pose_landmarks)

                    if skeleton_img is not None:
                        out_path = os.path.join(
                            skeleton_dir, f"{frame_id:06d}.png"
                        )
                        cv2.imwrite(out_path, skeleton_img)
                        skeleton_saved += 1

                # ---------- FACE → ALIGNED CROP ----------
                if face_landmarks is not None:
                    face_crop = aligned_face_crop(frame, face_landmarks)

                    if face_crop is not None:
                        out_path = os.path.join(
                            face_dir, f"{frame_id:06d}.png"
                        )
                        cv2.imwrite(out_path, face_crop)
                        face_saved += 1

            # ---------- SAVE QUALITY MASK ----------
            quality_path = os.path.join(quality_dir, "quality.json")
            with open(quality_path, "w") as f:
                json.dump(quality_list, f, indent=2)

            print(f"Skeleton frames saved: {skeleton_saved}")
            print(f"Face crops saved: {face_saved}")
            print(f"Quality entries saved: {len(quality_list)}")


if __name__ == "__main__":
    main()
