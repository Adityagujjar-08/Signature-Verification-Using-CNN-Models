# CNN-Based Offline Signature Verification for Document Image Authentication

Offline signature verification on the **CEDAR Signatures** dataset, framed as image-level
binary classification of each signature as **genuine** or **forged**. The project compares a
custom CNN baseline against MobileNetV2 and ResNet50 transfer-learning models (frozen and
fine-tuned), and adds Grad-CAM and SHAP explainability.

- **Student / Repository owner:** Aditya Patil
- **Dataset:** [CEDAR Signatures](https://www.kaggle.com/datasets/matteocarnebella/cedar-signatures)
- **Repository:** https://github.com/Adityagujjar-08/Signature-Verification-Using-CNN-Models
- **Framework:** TensorFlow / Keras + scikit-learn, developed in a Kaggle Notebook

> **Scope note.** This is single-image binary classification, suitable for a course CNN
> assignment. A production biometric system would use a pair-based (e.g. Siamese) approach that
> compares a questioned signature against reference genuine signatures from the claimed writer.

## Dataset

| Item | Value |
| --- | --- |
| Total images | 2,640 |
| Genuine / Forged | 1,320 / 1,320 (balanced) |
| Writers detected | 55 |
| Input size | 224 × 224 × 3 (RGB) |
| Split method | Writer-independent group split |

Train / validation / test = 1,824 / 384 / 432 images across 38 / 8 / 9 **disjoint** writers.
No writer appears in more than one split, which is enforced by an explicit overlap assertion in
the split cell.

## How to run

1. Open the notebook on Kaggle and attach the CEDAR dataset at
   `/kaggle/input/datasets/matteocarnebella/cedar-signatures`.
2. Run all cells. The notebook locates the dataset automatically, indexes and labels images,
   builds the writer-independent split, trains all models, evaluates them, and produces every
   figure, CSV, and model file.
3. Outputs are written to `/kaggle/working/signature_verification_outputs/` and bundled into
   `signature_verification_outputs.zip`.

Key configuration flags (top of the notebook): `FAST_DEBUG`, `RUN_FINE_TUNING`, `RUN_GRADCAM`,
`RUN_SHAP`, `SAVE_MODELS`, and the data-side regularization knobs (`PIPELINE_AUGMENT`,
`PIPELINE_CROP_FRACTION`, `PIPELINE_JPEG_QUALITY`, `PIPELINE_AUGMENT_VAL`, etc.).

## Models

| Model | Role | Total params | Trainable params |
| --- | --- | --- | --- |
| Custom CNN | Baseline trained from scratch | 422,337 | 421,889 |
| MobileNetV2 (fine-tuned) | Lightweight transfer learning | 2,422,081 | 1,674,817 |
| ResNet50 (fine-tuned) | Deeper transfer learning | 24,112,513 | 9,443,841 |

All models use a `Dense(1, sigmoid)` head, binary cross-entropy loss, and the Adam optimizer.
In-graph augmentation (rotation, translation, zoom, contrast) is combined with additional
**data-side** augmentation applied inside the `tf.data` pipeline (random crop, brightness /
contrast jitter, JPEG-quality degradation, Gaussian noise, blur, and cutout) so reported scores
reflect generalization to unseen writers rather than memorization.

## Results (test set, writer-independent)

| Model | Accuracy | Precision | Recall | F1 | AUC | EER | Train time (s) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Custom CNN | 0.6829 | 0.6138 | 0.9861 | 0.7567 | 0.8139 | 0.2546 | 224.37 |
| MobileNetV2 (fine-tuned) | 0.7407 | 0.7549 | 0.7130 | 0.7333 | 0.8317 | 0.2569 | 60.56 |
| ResNet50 (fine-tuned) | 0.7639 | 1.0000 | 0.5278 | 0.6909 | 0.9845 | 0.0648 | 90.15 |

Precision / recall / F1 are reported for the forged class.

**Reading the table.** ResNet50 has the highest accuracy and a much stronger AUC (0.98), i.e. it
ranks genuine vs. forged very well, but at the default 0.5 threshold it leans heavily toward
precision and misses nearly half of the forgeries (recall 0.53). MobileNetV2 gives the most
balanced operating point. The Custom CNN over-predicts the forged class, giving high recall but
low precision. The gap between ResNet50's AUC and its threshold accuracy is itself a useful
finding: a strong ranker can still be poorly calibrated at a fixed cutoff, which matters for a
verification system where the FAR/FRR trade-off must be chosen deliberately.

## Output structure

```
signature_verification_outputs/
├── csv/
│   ├── dataset_index.csv, dataset_description.csv, dataset_split_summary.csv
│   ├── train_split.csv, validation_split.csv, test_split.csv
│   ├── model_results_initial.csv, model_results_final.csv
│   ├── confusion_matrices_*.csv, classification_reports_*.csv
│   ├── per_sample_predictions_*.csv, error_analysis_misclassified_samples.csv
│   ├── model_architecture_table.csv, training_history.csv
│   ├── final_comparison_table.csv, research_questions.csv
│   └── fixed_shap_cam_statistics.csv, fixed_shap_cam_generation_status.csv
├── figures_pdf/
│   ├── figure_1_dataset_samples.pdf
│   ├── figure_2_model_design_workflow.pdf
│   ├── figure_3_training_curves.pdf
│   ├── figure_4_confusion_matrix_comparison.pdf
│   ├── figure_5_gradcam_correct_predictions.pdf
│   ├── figure_6_gradcam_misclassified_predictions.pdf
│   ├── figure_7_fixed_shap_cam_examples_Custom_CNN.pdf (+ .png)
│   ├── figure_8_final_model_comparison.pdf
│   └── figure_9_roc_curves.pdf
├── models/
│   ├── Custom_CNN.keras
│   ├── MobileNetV2_FineTuned.keras
│   └── ResNet50_FineTuned.keras
└── text_reports/
    ├── <Model>_summary.txt
    └── auto_report_summary.txt
```

## Evaluation metrics

Accuracy, precision, recall, F1, and AUC are reported alongside verification-specific metrics:
False Acceptance Rate (forged accepted as genuine), False Rejection Rate (genuine rejected as
forged), and Equal Error Rate. Confusion matrices, per-sample predictions, and a misclassified-
sample error-analysis CSV are saved for every final model.

## Explainability

- **Grad-CAM** highlights the image regions driving each prediction, for both correct and
  misclassified samples (Figures 5–6).
- **SHAP** (GradientExplainer on a logit version of the model) estimates positive and negative
  pixel evidence for the Custom CNN (Figure 7).

## Research questions

1. How accurately does a custom CNN classify offline signatures as genuine or forged on CEDAR?
2. Does transfer learning (MobileNetV2, ResNet50) improve over the custom CNN?
3. Does fine-tuning improve on frozen transfer learning?
4. Which model offers the best trade-off between performance, training time, and trainable size?
5. Which signature regions do the models focus on for correct predictions (Grad-CAM)?
6. What positive/negative visual evidence does SHAP reveal?
7. Can Grad-CAM and SHAP help explain misclassifications?

## Limitations

CEDAR is a relatively small dataset, and single-image classification is a simplification of true
reference-based verification. Genuine and forged CEDAR signatures also carry global differences
that a CNN can exploit, which is why AUC can stay high even under a clean writer-independent
split. Results should be read with this in mind, and a Siamese / pair-based extension (optionally
on a larger corpus such as BHSig260) is the natural next step.

## Reproducibility

A fixed random seed (`SEED = 42`) is set for Python, NumPy, and TensorFlow. The writer-independent
split, class weights, and all augmentation are seeded so runs are repeatable up to GPU
non-determinism.
