# Project SHIELD

**Secure Homomorphic Inference for Encrypted Learning on Drones**

Project SHIELD is a privacy-preserving AI framework designed to demonstrate real-time encrypted facial classification on drones using Fully Homomorphic Encryption (FHE). The system combines Singular Value Decomposition (SVD) for dimensionality reduction with encrypted inference to reduce latency and payload size while maintaining strong classification performance.

This repository presents both a raw-pixel baseline and an SVD-optimized variant of the model for performance comparison.

---

## Overview

### Objective

To explore whether a drone can detect potential threats in a crowd without exposing the identity or sensitive information of every individual. By integrating SVD with FHE, the system reduces computational complexity, enabling real-time encrypted inference.

### Key Concepts

- **Fully Homomorphic Encryption (FHE):** Enables computation on encrypted data without ever decrypting it. Data privacy is preserved end-to-end.
- **Singular Value Decomposition (SVD):** Reduces the feature dimensionality of facial images, allowing faster inference and lower encrypted payloads.

---

## Directory Structure

```
.
├── final_model_with_svd.py           # Optimized model using SVD features
├── final_model_no_svd.py             # Baseline model using raw pixels
├── final_results/                    # Output visualizations from SVD model
├── results/                          # Output visualizations from baseline model
├── dataset/
│   ├── Criminal/                     # Criminal-class labeled images
│   └── General/                      # General-class labeled images
├── base.jpg                          # Image used for simulated drone perturbation
├── sample_group.jpg                  # Image used for output annotation
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
```

---

## Experimental Summary

| Metric                  | SVD Model (k=8) | No-SVD Baseline | Change         |
|------------------------|------------------|------------------|----------------|
| Accuracy               | 0.7542           | 0.8475           | -11%           |
| Average Inference Time | 0.1192 sec       | 1.0676 sec       | -89%           |
| Payload Size           | 2,056 features   | 16,384 features  | -87%           |

Using SVD significantly reduces payload and latency while maintaining acceptable classification accuracy.

---

## Setup

### Requirements

Install dependencies using:

```bash
pip install -r requirements.txt
```

### Dataset Structure

Organize the dataset in the following structure:

```
dataset/
├── Criminal/
│   └── *.jpg or *.png
└── General/
    └── *.jpg or *.png
```

Ensure that the images are of frontal faces and that Haar cascades can detect them.

---

## Running the Code

### 1. Baseline (Raw Pixels)

```bash
python final_model_no_svd.py
```

Generates:
- Encrypted classification using raw pixel features
- Performance metrics and logs in `results/`
- Annotated visual output of predictions

### 2. Optimized Model (SVD)

```bash
python final_model_with_svd.py
```

Generates:
- Inference experiments for multiple SVD ranks
- Side-by-side perturbation visuals
- Trade-off plots for accuracy vs inference time and payload
- Annotated output images in `final_results/`

**Note:** Run the baseline script first to generate metrics used for plotting in the SVD script.

---

## Visual Outputs

The model produces the following outputs:
- `svd_compression_showcase.png`: Comparison of different SVD ranks
- `group_analysis_SVD_STANDARD.jpg`: Predictions on clean sample image
- `group_analysis_SVD_CHALLENGED_OUTPUT.jpg`: Predictions on perturbed sample image
- `accuracy_vs_inference_time_tradeoff.png`: Trade-off between speed and accuracy
- `accuracy_vs_payload_tradeoff.png`: Trade-off between payload size and accuracy

---

## Use Case and Relevance

This project demonstrates a viable pipeline for privacy-preserving surveillance systems using encrypted machine learning. Potential applications include:

- Drone-based crowd monitoring at public events
- Encrypted video analysis for defense or law enforcement
- Privacy-focused edge AI systems

---

## Future Work

- Extend from binary classification to facial identification
- Implement encrypted similarity search on embeddings
- Integrate with real-time drone camera streams
- Experiment with non-linear encrypted models

---

## References

- CelebA, iDOC Mugshots, and VGGFace2 datasets
- Concrete ML: https://docs.zama.ai/concrete-ml
- Visual SVD references: A. Kadhim (2022), S. Kahu (2013), Y. Wang (2017)
- Homomorphic inference literature: J. Frery et al., ICIS 2023

---

## Acknowledgements

This work was completed under the NSF-funded SCALE REU program.  
We thank Dr. Camp and Dr. Aceves for their mentorship and support.
