import pickle
import sys
from pathlib import Path
import cv2
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1
from huggingface_hub import hf_hub_download
from PIL import Image
from pillow_heif import register_heif_opener
from ultralytics import YOLO

# Register HEIF/HEIC support with Pillow
register_heif_opener()

# config
PHOTOS_DIR = Path("../Photos")
EMBEDDINGS_FILE = "embeddings_facenet.pkl"
YOLO_REPO = "arnabdhar/YOLOv8-Face-Detection"
YOLO_FILE = "model.pt"
FACENET_INPUT_SIZE = 160
IMAGE_EXTENSIONS = {".heic", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_models():
    """Load YOLO face-detector and FaceNet embedding model (CPU)."""
    yolo_path = hf_hub_download(repo_id=YOLO_REPO, filename=YOLO_FILE)
    yolo = YOLO(yolo_path)

    facenet = InceptionResnetV1(pretrained="vggface2").eval()
    print("[INFO] Models loaded (CPU mode).")
    return yolo, facenet


def detect_and_crop_face(yolo: YOLO, image: Image.Image):
    """Detect the largest face in an image and return the cropped region.

    Returns None if no face is detected.
    """
    img_rgb = np.array(image.convert("RGB"))
    results = yolo.predict(img_rgb, verbose=False)

    best_box = None
    best_area = 0
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        return None

    x1, y1, x2, y2 = best_box
    face_crop = img_rgb[y1:y2, x1:x2]
    return face_crop


def preprocess_face(face_crop: np.ndarray) -> torch.Tensor:
    """Resize to 160×160, normalise to [-1, 1], add batch dim."""
    face = cv2.resize(face_crop, (FACENET_INPUT_SIZE, FACENET_INPUT_SIZE))
    face = face.astype(np.float32) / 255.0  # [0, 1]
    face = (face - 0.5) / 0.5  # [-1, 1]
    tensor = torch.from_numpy(face).permute(2, 0, 1)  # (C, H, W)
    return tensor.unsqueeze(0)  # (1, C, H, W)


@torch.no_grad()
def get_embedding(facenet: InceptionResnetV1, face_tensor: torch.Tensor) -> np.ndarray:
    """Return L2-normalised 512-d embedding."""
    emb = facenet(face_tensor).squeeze().numpy()
    emb = emb / np.linalg.norm(emb)
    return emb


def enroll():
    if not PHOTOS_DIR.exists():
        print(f"[ERROR] Photos directory not found: {PHOTOS_DIR}")
        sys.exit(1)

    yolo, facenet = load_models()

    embeddings: dict[str, np.ndarray] = {}
    person_dirs = sorted(
        [d for d in PHOTOS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
    )

    if not person_dirs:
        print("[ERROR] No person subdirectories found in Photos/.")
        sys.exit(1)

    for person_dir in person_dirs:
        name = person_dir.name
        image_files = [
            f for f in person_dir.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_files:
            print(f"  [WARN] {name}: no images found — skipping.")
            continue

        person_embeddings = []
        for img_path in image_files:
            try:
                image = Image.open(img_path)
            except Exception as e:
                print(f"  [WARN] {name}: cannot open {img_path.name} — {e}")
                continue

            face_crop = detect_and_crop_face(yolo, image)
            if face_crop is None:
                print(f"  [WARN] {name}: no face in {img_path.name} — skipping.")
                continue

            face_tensor = preprocess_face(face_crop)
            emb = get_embedding(facenet, face_tensor)
            person_embeddings.append(emb)

        if person_embeddings:
            avg = np.mean(person_embeddings, axis=0)
            avg = avg / np.linalg.norm(avg)  # re-normalise
            embeddings[name] = avg
            print(f"  [OK] {name}: enrolled with {len(person_embeddings)} face(s)")
        else:
            print(f"  [WARN] {name}: no valid faces found — not enrolled.")

    if not embeddings:
        print("[ERROR] No faces enrolled. Check your Photos/ folder.")
        sys.exit(1)

    with open(EMBEDDINGS_FILE, "wb") as f:
        pickle.dump(embeddings, f)

    print(f"\n[INFO] Saved {len(embeddings)} person(s) to {EMBEDDINGS_FILE}")


if __name__ == "__main__":
    enroll()
