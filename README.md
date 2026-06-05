# CNN-Based Offline Signature Verification for Document Image Authentication

This repository contains a Kaggle notebook for offline handwritten signature verification using deep learning. The project compares a custom CNN with transfer-learning models and uses explainable AI methods to understand model predictions.

## Project Objective

The aim of this project is to classify signature images as either **genuine** or **forged**. The work follows a neural-network image classification workflow and includes:

- Custom CNN model
- MobileNetV2 transfer learning and fine-tuning
- ResNet50 transfer learning and fine-tuning
- Model evaluation using standard classification metrics
- Confusion matrix analysis
- Training and validation curves
- Grad-CAM visual explanations
- SHAP-based explanation outputs
- CSV result tables and PDF figures generated automatically

## Dataset

**Dataset:** CEDAR Signatures  
**Kaggle path used in notebook:**

```text
/kaggle/input/datasets/matteocarnebella/cedar-signatures
```

The dataset is suitable for this topic because it contains offline handwritten signature images labelled as genuine and forged. It is appropriate for binary CNN-based image classification and document authentication experiments.

Dataset summary generated from the notebook:

| Item | Value |
|---|---:|
| Total images | 2640 |
| Genuine images | 1320 |
| Forged images | 1320 |
| Detected writers | 55 |
| Image input size | 224 x 224 x 3 |
| Classification type | Binary image classification |

## Dataset Split

The notebook uses a writer-independent group split to reduce writer leakage between training, validation, and testing sets.

| Split | Images | Genuine | Forged | Writers |
|---|---:|---:|---:|---:|
| Train | 1824 | 912 | 912 | 38 |
| Validation | 384 | 192 | 192 | 8 |
| Test | 432 | 216 | 216 | 9 |

## Models Implemented

| Model | Input Size | Output Layer | Loss Function | Optimizer |
|---|---|---|---|---|
| Custom CNN | 224 x 224 x 3 | Dense(1, sigmoid) | Binary crossentropy | Adam |
| MobileNetV2 Fine-Tuned | 224 x 224 x 3 | Dense(1, sigmoid) | Binary crossentropy | Adam |
| ResNet50 Fine-Tuned | 224 x 224 x 3 | Dense(1, sigmoid) | Binary crossentropy | Adam |

The notebook also performs transfer learning before fine-tuning. The final comparison is reported for the fine-tuned models.

## Evaluation Metrics

The following metrics are generated and saved in CSV files:

- Accuracy
- Precision for forged class
- Recall for forged class
- F1-score for forged class
- AUC
- FAR: False Acceptance Rate
- FRR: False Rejection Rate
- EER: Equal Error Rate
- Training time
- Total parameters
- Trainable parameters
- Non-trainable parameters

## Final Model Results

| Model | Accuracy | Precision | Recall | F1-score | AUC | EER | Training Time |
|---|---:|---:|---:|---:|---:|---:|---:|
| Custom CNN | 0.9606 | 1.0000 | 0.9213 | 0.9590 | 1.0000 | 0.0000 | 94.70 s |
| MobileNetV2 Fine-Tuned | 0.7222 | 1.0000 | 0.4444 | 0.6154 | 0.9686 | 0.0949 | 44.80 s |
| ResNet50 Fine-Tuned | 0.9699 | 1.0000 | 0.9398 | 0.9690 | 0.9999 | 0.0046 | 59.10 s |

The best model by F1-score is **ResNet50 Fine-Tuned**.

## Explainable AI

The notebook generates visual explanations using:

### Grad-CAM

Grad-CAM is used to highlight image regions that influenced the model prediction. The notebook creates visual examples for:

- Correct predictions
- Misclassified predictions
- Model comparison across Custom CNN, MobileNetV2, and ResNet50

### SHAP

SHAP is used to estimate positive and negative visual evidence for selected test images. SHAP can be slow on Kaggle, so the notebook saves status files if SHAP generation fails due to runtime or package limitations.

## Generated Output Structure

After running the notebook on Kaggle, the output ZIP is created at:

```text
/kaggle/working/signature_verification_outputs.zip
```

The ZIP contains:

```text
signature_verification_outputs/
├── csv/
│   ├── dataset_description.csv
│   ├── dataset_split_summary.csv
│   ├── model_architecture_table.csv
│   ├── model_results_initial.csv
│   ├── model_results_final.csv
│   ├── final_comparison_table.csv
│   ├── classification_reports_final.csv
│   ├── confusion_matrices_final.csv
│   ├── training_history.csv
│   ├── per_sample_predictions_final.csv
│   └── research_questions.csv
│
├── figures_pdf/
│   ├── figure_1_dataset_samples.pdf
│   ├── figure_2_model_design_workflow.pdf
│   ├── figure_3_training_curves.pdf
│   ├── figure_4_confusion_matrix_comparison.pdf
│   ├── figure_5_gradcam_correct_predictions.pdf
│   ├── figure_6_gradcam_misclassified_predictions.pdf
│   ├── figure_7_fixed_shap_cam_examples_Custom_CNN.pdf
│   ├── figure_8_final_model_comparison.pdf
│   └── figure_9_roc_curves.pdf
│
├── models/
│   ├── Custom_CNN.keras
│   ├── MobileNetV2_FineTuned.keras
│   └── ResNet50_FineTuned.keras
│
└── text_reports/
    ├── auto_report_summary.txt
    ├── Custom_CNN_summary.txt
    ├── MobileNetV2_FineTuned_summary.txt
    └── ResNet50_FineTuned_summary.txt
```

## How to Run on Kaggle

1. Open Kaggle Notebooks.
2. Add the CEDAR Signatures dataset to the notebook input.
3. Make sure the dataset is available at:

```text
/kaggle/input/datasets/matteocarnebella/cedar-signatures
```

4. Upload or open the notebook:

```text
ml-cnn-signature-verification.ipynb
```

5. Run all cells.
6. Download the generated output ZIP from:

```text
/kaggle/working/signature_verification_outputs.zip
```

## Repository Files

Recommended GitHub repository structure:

```text
.
├── README.md
├── ml-cnn-signature-verification.ipynb
├── signature_verification_outputs.zip
└── requirements.txt  # optional
```

## Important Notes

This project performs **single-image binary classification** of signatures as genuine or forged. It is suitable for a neural-network image classification assignment, but it is not a full production-level signature verification system.

A real signature verification system usually compares a questioned signature against one or more reference signatures from the same writer. The current model learns from labelled images directly, so high scores may still be influenced by dataset-specific visual patterns such as scanning style, background, stroke thickness, or image formatting.

For a stronger research version, the next improvements should include:

- Pair-based or triplet-based signature verification
- Siamese networks
- More datasets such as BHSig260 or GPDS
- Stronger writer-independent testing
- External dataset validation
- More detailed FAR, FRR, and EER analysis

## Research Questions

1. How accurately does a custom CNN classify offline signature images as genuine or forged?
2. Does transfer learning with MobileNetV2 and ResNet50 improve signature verification performance compared with a custom CNN?
3. Does fine-tuning improve performance compared with frozen transfer learning models?
4. Which model provides the best trade-off between performance, training time, and trainable parameters?
5. Which signature regions do the CNN models focus on during correct predictions according to Grad-CAM?
6. What do SHAP explanations reveal about positive and negative visual evidence in the signature images?
7. Can Grad-CAM and SHAP help explain why some signature images are misclassified?

## License

This repository is prepared for academic coursework. Check the original dataset license and Kaggle terms before redistributing the dataset or trained models.

## Author

Prepared for the topic:

**CNN-Based Offline Signature Verification for Document Image Authentication**
