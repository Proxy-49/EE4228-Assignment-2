import pickle
import time
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1

# config
EMBEDDINGS_FILE = Path(__file__).resolve().parent / "embeddings_facenet.pkl"

FACENET_INPUT_SIZE = 160
DETECTION_CONFIDENCE = 0.5
RECOGNITION_THRESHOLD = 0.6  # cosine-similarity threshold

BOX_COLOR = (0, 255, 0)  # BGR green
UNKNOWN_BOX_COLOR = (0, 0, 255)  # BGR red for unknown faces
BOX_THICKNESS = 2
LABEL_FONT_SCALE = 0.6
CAMERA_INDEX = 1


# model loading
def load_mtcnn() -> MTCNN:
    return MTCNN(keep_all=True, device="cpu")


def load_facenet() -> InceptionResnetV1:
    model = InceptionResnetV1(pretrained="vggface2").eval()
    return model


def load_embeddings() -> dict[str, np.ndarray]:
    if not EMBEDDINGS_FILE.exists():
        print(f"[ERROR] Embeddings not found at {EMBEDDINGS_FILE}")
        print("        Run  enroll.py  first to generate them.")
        sys.exit(1)
    with open(EMBEDDINGS_FILE, "rb") as f:
        data = pickle.load(f)
    print(f"[INFO] Loaded {len(data)} enrolled person(s).")
    return data


# face processing
def preprocess_face(face_bgr: np.ndarray) -> torch.Tensor:
    """Resize, normalise to [-1, 1], return (1, 3, 160, 160) tensor."""
    face = cv2.resize(face_bgr, (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face.astype(np.float32) / 255.0
    face = (face - 0.5) / 0.5
    tensor = torch.from_numpy(face).permute(2, 0, 1)
    return tensor.unsqueeze(0)


@torch.no_grad()
def get_embedding(facenet: InceptionResnetV1, face_tensor: torch.Tensor) -> np.ndarray:
    emb = facenet(face_tensor).squeeze().numpy()
    return emb / np.linalg.norm(emb)


def identify(
    embedding: np.ndarray,
    enrolled: dict[str, np.ndarray],
) -> tuple[str, float]:
    """Return (name, similarity) of the best match, or ("Unknown", sim)."""
    best_name = "Unknown"
    best_sim = -1.0
    for name, ref_emb in enrolled.items():
        sim = float(np.dot(embedding, ref_emb))  # cosine (both L2-normed)
        if sim > best_sim:
            best_sim = sim
            best_name = name
    if best_sim < RECOGNITION_THRESHOLD:
        return "Unknown", best_sim
    return best_name, best_sim


# drawing
def draw_box(frame, x1, y1, x2, y2, label: str, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICKNESS)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, LABEL_FONT_SCALE, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        LABEL_FONT_SCALE,
        (0, 0, 0),
        1,
        cv2.LINE_AA,
    )


# main loop
def main():
    mtcnn = load_mtcnn()
    facenet = load_facenet()
    enrolled = load_embeddings()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return

    print("[INFO] Press 'q' to quit.")

    prev_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[WARN] Failed to grab frame — exiting.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1.0 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Convert BGR → RGB for MTCNN
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, confidences = mtcnn.detect(rgb_frame)

        if boxes is not None:
            for box, conf in zip(boxes, confidences):
                if conf is None or conf < DETECTION_CONFIDENCE:
                    continue

                x1, y1, x2, y2 = map(int, box)

                # Clamp to frame bounds
                h, w = frame.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)

                face_crop = frame[y1:y2, x1:x2]
                if face_crop.size == 0:
                    continue

                face_tensor = preprocess_face(face_crop)
                emb = get_embedding(facenet, face_tensor)
                name, sim = identify(emb, enrolled)

                label = f"{name} ({sim:.2f})"
                color = BOX_COLOR if name != "Unknown" else UNKNOWN_BOX_COLOR
                draw_box(frame, x1, y1, x2, y2, label, color)

        # Draw FPS in top-left corner
        cv2.putText(
            frame,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("MTCNN + FaceNet Face Recognition", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Done.")


if __name__ == "__main__":
    main()
