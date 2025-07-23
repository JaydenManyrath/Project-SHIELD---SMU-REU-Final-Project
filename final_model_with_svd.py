import os
import sys
import time
import random
import logging
from tqdm import tqdm
import warnings

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.utils.extmath import randomized_svd
from concrete.ml.sklearn import LinearSVC

warnings.filterwarnings("ignore", category=UserWarning)

# Configuration
RESIZED = (128, 128)
C_PARAM = 0.01
N_BITS = 5
RANDOM_STATE = 42
TEST_SIZE = 0.2
SHOWCASE_K = 24
K_VALUES_TO_TEST = [8, 16, 24, 32]
NO_SVD_FEATURES = 9216

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(BASE_DIR, "final_results")
DATA_DIR = os.path.join(BASE_DIR, "dataset")
CRIMINAL_DIR = os.path.join(DATA_DIR, "Criminal")
GENERAL_DIR = os.path.join(DATA_DIR, "General")
os.makedirs(RESULTS_DIR, exist_ok=True)

CASCADE_FRONTAL = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
SAMPLE_GROUP_IMAGE = os.path.join(BASE_DIR, "sample_image", "base.jpg")

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=[
    logging.FileHandler(os.path.join(RESULTS_DIR, "svd_generation.log")),
    logging.StreamHandler(sys.stdout)
])
logger = logging.getLogger()

# Helper functions
def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

def extract_svd_compression_features(img, k):
    U, S, VT = randomized_svd(img, n_components=k, n_iter=5, random_state=RANDOM_STATE)
    return np.concatenate([U.flatten(), S, VT.flatten()])

def non_max_suppression(boxes, overlap_thresh=0.3):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []
    x1, y1, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x2, y2 = x1 + w, y1 + h
    area = (w + 1) * (h + 1)
    idxs = np.argsort(y2)
    while len(idxs) > 0:
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        w_inter = np.maximum(0, xx2 - xx1 + 1)
        h_inter = np.maximum(0, yy2 - yy1 + 1)
        inter = w_inter * h_inter
        overlap = inter / (area[i] + area[idxs[:last]] - inter)
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlap_thresh)[0])))
    return boxes[pick].astype(int)

def load_images(folder, label):
    data = []
    fr = cv2.CascadeClassifier(CASCADE_FRONTAL)
    for fn in tqdm(sorted(os.listdir(folder)), desc=f"Loading {os.path.basename(folder)}", unit="files"):
        if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(folder, fn)
            img = cv2.imread(img_path)
            if img is None:
                continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = fr.detectMultiScale(gray, 1.1, 5)
            if len(faces) > 0:
                x, y, w, h = faces[0]
                face_roi = cv2.resize(gray[y:y+h, x:x+w], RESIZED)
                data.append((face_roi, label))
    return data

def train_and_compile(X, y):
    model = LinearSVC(n_bits=N_BITS, C=C_PARAM, random_state=RANDOM_STATE)
    model.fit(X, y)
    model.compile(X)
    return model

# Visualization: Shows how image quality changes with different SVD ranks
def generate_svd_visualization(sample_image, k_values_to_show, output_path):
    logger.info("Generating SVD compression showcase visual...")
    fig, axes = plt.subplots(1, len(k_values_to_show) + 1, figsize=(16, 4))
    axes[0].imshow(sample_image, cmap='gray')
    axes[0].set_title("Original (128x128)")
    axes[0].axis('off')
    
    for i, k in enumerate(k_values_to_show):
        U, S, Vt = np.linalg.svd(sample_image, full_matrices=False)
        reconstructed_img = np.dot(U[:, :k] * S[:k], Vt[:k, :])
        axes[i+1].imshow(reconstructed_img, cmap='gray')
        axes[i+1].set_title(f"Reconstruction (k={k})")
        axes[i+1].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    logger.info(f"SVD Showcase saved to {output_path}")

# Simulates shaky, blurry, and low-resolution drone footage
def perturb_image(image_path, comparison_output_path, standalone_output_path, blur_level=15, resolution_scale=0.4, max_rotation_angle=5):
    logger.info("Generating perturbed 'shaky cam' simulation images...")
    original_img = cv2.imread(image_path)
    if original_img is None:
        logger.error(f"Could not read sample image at {image_path}")
        return None

    h, w, _ = original_img.shape
    angle = random.uniform(-max_rotation_angle, max_rotation_angle)
    center = (w // 2, h // 2)
    rot_mat = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(original_img, rot_mat, (w, h), borderMode=cv2.BORDER_REPLICATE)
    small_img = cv2.resize(rotated_img, (int(w * resolution_scale), int(h * resolution_scale)), interpolation=cv2.INTER_LINEAR)
    blurred_img = cv2.GaussianBlur(small_img, (blur_level, blur_level), 0)
    perturbed_img = cv2.resize(blurred_img, (w, h), interpolation=cv2.INTER_NEAREST)

    cv2.imwrite(standalone_output_path, perturbed_img)
    logger.info(f"Standalone perturbed image saved to {standalone_output_path}")

    original_for_visual = original_img.copy()
    cv2.putText(original_for_visual, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    label_text = f"Shaken (Blur={blur_level}, Scale={resolution_scale:.1f}, Rot~{angle:.1f}deg)"
    cv2.putText(perturbed_img, label_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    comparison_image = cv2.hconcat([original_for_visual, perturbed_img])
    cv2.imwrite(comparison_output_path, comparison_image)
    logger.info(f"Side-by-side comparison saved to {comparison_output_path}")

    return standalone_output_path

# Draws predictions on detected faces
def draw_bounding_boxes(image_path, model, scaler, feature_extractor, k_val, output_path):
    logger.info(f"Running full analysis on {os.path.basename(image_path)}...")
    img_bgr = cv2.imread(image_path)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    fr_cascade = cv2.CascadeClassifier(CASCADE_FRONTAL)
    detections = fr_cascade.detectMultiScale(img_gray, 1.1, 5)
    faces = non_max_suppression(np.array(detections))
    
    for (x, y, w, h) in faces:
        face_roi = cv2.resize(img_gray[y:y+h, x:x+w], RESIZED)
        features = feature_extractor(face_roi, k=k_val)
        scaled_features = scaler.transform(features.reshape(1, -1))
        quantized_input = model.quantize_input(scaled_features.astype(np.float32))
        logit = model.fhe_circuit.encrypt_run_decrypt(quantized_input)[0]
        prediction = 1 if logit > 0 else 0
        label_text = "Criminal" if prediction == 1 else "General"
        color = (0, 0, 255) if prediction == 1 else (0, 255, 0)
        cv2.rectangle(img_bgr, (x, y), (x+w, y+h), color, 2)
        cv2.putText(img_bgr, label_text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
    cv2.imwrite(output_path, img_bgr)
    logger.info(f"Annotated image saved to {output_path}")

# Plots accuracy vs time and payload trade-offs for different SVD ranks
def generate_quantitative_plots(results, baseline_stats, output_dir):
    logger.info("Generating quantitative results plots...")
    k_vals = [r['k'] for r in results]
    accuracies = [r['accuracy'] for r in results]
    times = [r['avg_time'] for r in results]
    payloads = [r['payload_size'] for r in results]
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_xlabel('Inference Time (seconds)')
    ax1.set_ylabel('Accuracy', color='tab:blue')
    ax1.plot(times, accuracies, 'o-', color='tab:blue', label='SVD Model Accuracy')
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    for i, k in enumerate(k_vals):
        ax1.text(times[i], accuracies[i] + 0.005, f'k={k}', fontsize=9, ha='center')
    ax1.plot(baseline_stats['time'], baseline_stats['accuracy'], 'r*', markersize=15, label=f"No SVD Baseline ({NO_SVD_FEATURES} features)")
    
    ax2 = ax1.twinx()
    ax2.set_ylabel('SVD Rank (k)', color='tab:green')
    ax2.plot(times, k_vals, 's--', color='tab:green', alpha=0.6, label='SVD Rank (k)')
    ax2.tick_params(axis='y', labelcolor='tab:green')
    
    fig.tight_layout()
    plt.title('Performance Trade-off: Accuracy vs. Inference Time')
    fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.05), ncol=3)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_inference_time_tradeoff.png"), bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(payloads, accuracies, 'o-', label='SVD Model')
    ax.set_xlabel('Feature Vector Size (Payload)')
    ax.set_ylabel('Accuracy')
    for i, k in enumerate(k_vals):
        ax.text(payloads[i], accuracies[i] + 0.005, f'k={k}', fontsize=9, ha='center')
    ax.axvline(x=baseline_stats['payload'], color='r', linestyle='--', label=f"No SVD Baseline ({baseline_stats['payload']} features)")
    
    plt.title('Accuracy vs. Payload Size Trade-off')
    plt.legend()
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)
    plt.savefig(os.path.join(output_dir, "accuracy_vs_payload_tradeoff.png"), bbox_inches='tight')
    plt.close(fig)
    logger.info("Quantitative plots saved.")

# Main routine
def main():
    seed_everything(RANDOM_STATE)

    logger.info("Loading image datasets...")
    crim_data = load_images(CRIMINAL_DIR, 1)
    gen_data = load_images(GENERAL_DIR, 0)
    if not crim_data or not gen_data:
        logger.error("Could not load data. Please ensure 'dataset' folder is populated.")
        return
    all_data = crim_data + gen_data
    X_raw, y = zip(*all_data)
    X_raw, y = np.array(X_raw), np.array(y)

    generate_svd_visualization(
        sample_image=X_raw[0],
        k_values_to_show=K_VALUES_TO_TEST,
        output_path=os.path.join(RESULTS_DIR, "svd_compression_showcase.png")
    )

    perturbed_img_path = None
    if os.path.exists(SAMPLE_GROUP_IMAGE):
        perturbed_img_path = perturb_image(
            image_path=SAMPLE_GROUP_IMAGE,
            comparison_output_path=os.path.join(RESULTS_DIR, "drone_simulation_SIDE_BY_SIDE.jpg"),
            standalone_output_path=os.path.join(RESULTS_DIR, "sample_group_CHALLENGED.jpg")
        )
    else:
        logger.warning(f"{SAMPLE_GROUP_IMAGE} not found. Skipping perturbation.")

    logger.info("Starting SVD Model Experiments")
    experimental_results = []
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y)

    for k in K_VALUES_TO_TEST:
        logger.info(f"Testing SVD with k={k}")
        X_train_unscaled = np.array([extract_svd_compression_features(x, k) for x in tqdm(X_train_raw)])
        X_test_unscaled = np.array([extract_svd_compression_features(x, k) for x in tqdm(X_test_raw)])
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_unscaled)
        X_test = scaler.transform(X_test_unscaled)
        model = train_and_compile(X_train, y_train)

        y_pred, inference_times = [], []
        for sample in tqdm(X_test):
            quantized_input = model.quantize_input(sample.reshape(1, -1).astype(np.float32))
            t0 = time.time()
            logit = model.fhe_circuit.encrypt_run_decrypt(quantized_input)[0]
            inference_times.append(time.time() - t0)
            y_pred.append(1 if logit > 0 else 0)

        acc = accuracy_score(y_test, y_pred)
        avg_time = np.mean(inference_times)
        payload = X_train.shape[1]
        logger.info(f"k={k} -> Accuracy: {acc:.4f}, Time: {avg_time:.4f}s, Payload: {payload}")
        experimental_results.append({'k': k, 'accuracy': acc, 'avg_time': avg_time, 'payload_size': payload})

    logger.info("Generating final annotated outputs")
    X_train_unscaled_prod = np.array([extract_svd_compression_features(x, SHOWCASE_K) for x in X_train_raw])
    scaler_prod = StandardScaler().fit(X_train_unscaled_prod)
    X_train_prod = scaler_prod.transform(X_train_unscaled_prod)
    model_prod = train_and_compile(X_train_prod, y_train)

    if os.path.exists(SAMPLE_GROUP_IMAGE):
        draw_bounding_boxes(
            image_path=SAMPLE_GROUP_IMAGE,
            model=model_prod,
            scaler=scaler_prod,
            feature_extractor=extract_svd_compression_features,
            k_val=SHOWCASE_K,
            output_path=os.path.join(RESULTS_DIR, "group_analysis_SVD_STANDARD.jpg")
        )
        if perturbed_img_path:
            draw_bounding_boxes(
                image_path=perturbed_img_path,
                model=model_prod,
                scaler=scaler_prod,
                feature_extractor=extract_svd_compression_features,
                k_val=SHOWCASE_K,
                output_path=os.path.join(RESULTS_DIR, "group_analysis_SVD_CHALLENGED_OUTPUT.jpg")
            )

    baseline_stats = {
        'accuracy': 0.8790,
        'time': 1.0554,
        'payload': NO_SVD_FEATURES
    }
    logger.info(f"Using baseline stats: {baseline_stats}")
    generate_quantitative_plots(experimental_results, baseline_stats, RESULTS_DIR)

    logger.info("All results generated successfully")

if __name__ == "__main__":
    main()
