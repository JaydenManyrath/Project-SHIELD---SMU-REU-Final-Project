import os
import sys
import time
import random
import logging
from tqdm import tqdm
import warnings

import cv2
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from concrete.ml.sklearn import LinearSVC

# Ignore benign warnings from libraries
warnings.filterwarnings("ignore", category=UserWarning)

# ─── CONFIG ───────────────────────────────────────────────────────────────────
RESIZED          = (128, 128)
C_PARAM          = 0.01
N_BITS           = 5
RANDOM_STATE     = 42
TEST_SIZE        = 0.2

# --- File & Directory Setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "results")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
CRIMINAL_DIR = os.path.join(DATA_DIR, "Criminal")
GENERAL_DIR = os.path.join(DATA_DIR, "General")
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Haar Cascades & Sample Image
CASCADE_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SAMPLE_GROUP_IMAGE = os.path.join(BASE_DIR, "sample_group.jpg")

# ─── LOGGING ──────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[
    logging.FileHandler(os.path.join(RESULTS_DIR, "baseline_generation.log")),
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger()

# ─── CORE HELPERS ─────────────────────────────────────────────────────────────
def seed_everything(seed):
    random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed); np.random.seed(seed)

def extract_raw_pixel_features(img):
    """Feature extraction is just flattening the raw image pixels."""
    return img.flatten()

def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0: return []
    if boxes.dtype.kind == "i": boxes = boxes.astype("float")
    pick = []; x1,y1,w,h = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]; x2,y2=x1+w,y1+h; area=(w+1)*(h+1); idxs=np.argsort(y2)
    while len(idxs) > 0:
        last=len(idxs)-1; i=idxs[last]; pick.append(i)
        xx1=np.maximum(x1[i],x1[idxs[:last]]); yy1=np.maximum(y1[i],y1[idxs[:last]]); xx2=np.minimum(x2[i],x2[idxs[:last]]); yy2=np.minimum(y2[i],y2[idxs[:last]])
        w_inter=np.maximum(0,xx2-xx1+1); h_inter=np.maximum(0,yy2-yy1+1); inter=w_inter*h_inter; overlap=inter/(area[i]+area[idxs[:last]]-inter)
        idxs=np.delete(idxs,np.concatenate(([last],np.where(overlap>overlap_thresh)[0])))
    return boxes[pick].astype(int)

def load_images(folder, label):
    data=[]; fr=cv2.CascadeClassifier(CASCADE_FRONTAL)
    for fn in tqdm(sorted(os.listdir(folder)), desc=f"Loading {os.path.basename(folder)}", unit="files"):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, fn)
            img = cv2.imread(img_path)
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = fr.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0] # Take the first detected face
                face_roi = cv2.resize(gray[y:y+h, x:x+w], RESIZED)
                data.append((face_roi, label))
    return data

def train_and_compile(X, y):
    model = LinearSVC(n_bits=N_BITS, C=C_PARAM, random_state=RANDOM_STATE)
    logger.info(f"\n--- Training Model ---\nTraining on {X.shape[0]} samples with {X.shape[1]} features each.")
    model.fit(X, y); logger.info("Compiling model to FHE circuit...")
    model.compile(X); logger.info("Training and compilation complete.")
    return model

# ─── VISUAL GENERATION FUNCTIONS ──────────────────────────────────────────────

def perturb_image(image_path, output_path, blur_level=5, resolution_scale=0.6):
    """Generates the 'drone simulation' image."""
    logger.info("Generating perturbed 'drone simulation' image...")
    img = cv2.imread(image_path)
    if img is None:
        logger.error(f"Could not read sample image at {image_path}"); return None

    h, w, _ = img.shape
    small_img = cv2.resize(img, (int(w * resolution_scale), int(h * resolution_scale)), interpolation=cv2.INTER_LINEAR)
    blurred_img = cv2.GaussianBlur(small_img, (blur_level, blur_level), 0)
    final_img = cv2.resize(blurred_img, (w, h), interpolation=cv2.INTER_NEAREST)
    
    cv2.imwrite(output_path, final_img)
    logger.info(f"✅ Perturbed image saved to {output_path}")
    return output_path

def draw_bounding_boxes(image_path, model, scaler, output_path):
    """Generates the annotated output images."""
    logger.info(f"Running full analysis on {os.path.basename(image_path)}...")
    img_bgr = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fr_cascade = cv2.CascadeClassifier(CASCADE_FRONTAL)
    
    detections = fr_cascade.detectMultiScale(img_gray, 1.1, 5)
    faces = non_max_suppression(np.array(detections))
    
    for (x, y, w, h) in faces:
        face_roi = cv2.resize(img_gray[y:y+h, x:x+w], RESIZED)
        features = extract_raw_pixel_features(face_roi)
        scaled_features = scaler.transform(features.reshape(1, -1))
        quantized_input = model.quantize_input(scaled_features.astype(np.float32))
        
        logit = model.fhe_circuit.encrypt_run_decrypt(quantized_input)[0]
        prediction = 1 if logit > 0 else 0
        
        label_text = "Criminal" if prediction == 1 else "General"
        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_bgr, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    cv2.imwrite(output_path, img_bgr)
    logger.info(f"✅ Annotated image saved to {output_path}")

# ─── MAIN EXECUTION ───────────────────────────────────────────────────────────
def main():
    seed_everything(RANDOM_STATE)
    
    # --- Generate Perturbed Image (Visual 2 from master list) ---
    if not os.path.exists(SAMPLE_GROUP_IMAGE):
        logger.warning(f"{SAMPLE_GROUP_IMAGE} not found. Skipping drone simulation visuals.")
        perturbed_img_path = None
    else:
        # Note: This is the same function as in the SVD script. It will overwrite
        # the file, which is fine as they are identical.
        perturbed_img_path = perturb_image(
            image_path=SAMPLE_GROUP_IMAGE,
            output_path=os.path.join(RESULTS_DIR, "sample_group_CHALLENGED.jpg")
        )

    # --- Load Data & Train Baseline Model ---
    logger.info("Loading image datasets...")
    crim_data = load_images(CRIMINAL_DIR, 1)
    gen_data = load_images(GENERAL_DIR, 0)
    if not crim_data or not gen_data:
        logger.error("Could not load data. Please ensure 'dataset' folder is populated. Exiting.")
        return
        
    all_data = crim_data + gen_data
    X_raw, y = zip(*all_data)
    X_raw, y = np.array(X_raw), np.array(y)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    
    logger.info("\n--- Applying Raw Pixel Feature Extraction (9216 features) ---")
    X_train_unscaled = np.array([extract_raw_pixel_features(x) for x in tqdm(X_train_raw, desc="Processing Train Set")])
    X_test_unscaled = np.array([extract_raw_pixel_features(x) for x in tqdm(X_test_raw, desc="Processing Test Set")])
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_unscaled)
    X_test = scaler.transform(X_test_unscaled)
    
    model = train_and_compile(X_train, y_train)

    logger.info("\n--- Evaluating Model on Test Set ---")
    y_pred, inference_times = [], []
    for sample in tqdm(X_test, desc="FHE Inference on Test Set"):
        xq = model.quantize_input(sample.reshape(1, -1).astype(np.float32))
        t0 = time.time()
        logit = model.fhe_circuit.encrypt_run_decrypt(xq)[0]
        t_fhe = time.time() - t0
        y_pred.append(1 if logit > 0 else 0)
        inference_times.append(t_fhe)

    # --- Print Final Quantitative Results ---
    accuracy = accuracy_score(y_test, y_pred)
    avg_time = np.mean(inference_times)
    
    logger.info("\n" + "="*60 + "\n           NO SVD BASELINE PERFORMANCE\n" + "="*60)
    logger.info(f"\n>>>> Overall Accuracy: {accuracy:.4f} <<<<")
    logger.info(f">>>> Average FHE Inference Time: {avg_time:.4f} seconds per sample <<<<")
    logger.info("\n(COPY THE TWO VALUES ABOVE INTO THE SVD SCRIPT FOR PLOTTING)\n")
    logger.info("Classification Report:\n" + classification_report(y_test, y_pred, target_names=["General (0)", "Criminal (1)"], zero_division=0))
    logger.info("\nNo SVD Baseline complete.")

    # --- Generate Annotated Images (Visual 3 from master list) ---
    if os.path.exists(SAMPLE_GROUP_IMAGE):
        logger.info("\n--- Generating Annotated Images for Baseline Model ---")
        draw_bounding_boxes(
            image_path=SAMPLE_GROUP_IMAGE, model=model, scaler=scaler,
            output_path=os.path.join(RESULTS_DIR, "group_analysis_BASELINE_STANDARD.jpg")
        )
        if perturbed_img_path:
            draw_bounding_boxes(
                image_path=perturbed_img_path, model=model, scaler=scaler,
                output_path=os.path.join(RESULTS_DIR, "group_analysis_BASELINE_CHALLENGED_OUTPUT.jpg")
            )

    logger.info("\nAll Baseline model visuals have been generated in the 'results' folder")

if __name__ == "__main__":
    main()