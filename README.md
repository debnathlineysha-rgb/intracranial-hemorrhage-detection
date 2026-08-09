# Intracranial Hemorrhage Detection — Binary CNN Classifier

A convolutional neural network that classifies brain CT scan images as either **normal** or **hemorrhagic**, built as a portfolio project exploring computer vision applications in medical imaging.

## Dataset

- **Source**: [Brain CT Hemorrhage Dataset](https://www.kaggle.com/datasets/abdulkader90/brain-ct-hemorrhage-dataset) by `abdulkader90` (Kaggle)
- **Size**: 6,795 total images — 2,690 hemorrhage, 4,105 normal
- **Structure**: Hemorrhage images are organized into per-patient folders (multiple CT slices per patient); normal images are in a flat folder.

## Pipeline

1. Images loaded via recursive `glob`, case-insensitive `.jpg` matching
2. Converted to grayscale, resized to 224×224
3. Loaded as `uint8` and only cast to `float32` (normalized 0–1) after the train/test split, to reduce peak memory usage during preprocessing
4. Reshaped to `(-1, 224, 224, 1)` for Keras
5. Train/test split: 80/20, stratified, `random_state=42`

## Model Architecture

A simple CNN built with Keras:

```
Conv2D(32) → MaxPooling2D
Conv2D(64) → MaxPooling2D
Conv2D(128) → MaxPooling2D
Flatten
Dense(128, relu)
Dropout(0.5)
Dense(1, sigmoid)
```

Compiled with the Adam optimizer and binary cross-entropy loss. Trained for 5 epochs.

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 99.85% |
| Precision (both classes) | 1.00 |
| Recall (both classes) | 1.00 |
| F1-score (both classes) | 1.00 |

## Known Limitation: Potential Data Leakage

The reported accuracy above is very likely **inflated by patient-level data leakage**. The hemorrhage images are organized into per-patient folders, each containing multiple CT slices from the same scan. Because the train/test split was performed at the **image level** rather than the **patient level**, slices from the same patient may appear in both the training and test sets. Since consecutive CT slices from one patient are visually very similar, the model may be partly recognizing patients it has already seen rather than learning fully generalizable features of hemorrhage detection.

**A more rigorous evaluation would split by patient ID**, ensuring all slices from a given patient fall entirely within either the training set or the test set, never both. This is a standard and well-known pitfall in medical imaging ML, and addressing it is a natural next step for this project — planned as a follow-up iteration.

Even accounting for this, the results suggest the model successfully learned relevant visual features and the pipeline (data loading, preprocessing, training) works correctly end-to-end.

## What This Project Demonstrates

- End-to-end image classification pipeline: raw data → preprocessing → model training → evaluation
- Practical debugging of real-world data issues (inconsistent folder structures, memory management for large image datasets)
- Awareness of methodological pitfalls specific to medical imaging ML (patient-level leakage), rather than just reporting a headline accuracy number

## Tech Stack

Python, TensorFlow/Keras, OpenCV, scikit-learn, matplotlib, seaborn — developed on Kaggle Notebooks (GPU-enabled).

## Future Improvements

- [ ] Re-split train/test by patient ID to eliminate leakage risk and get a more trustworthy accuracy figure
- [ ] Add data augmentation to improve robustness
- [ ] Try transfer learning with a pretrained model (e.g. ResNet, EfficientNet) for comparison
- [ ] Expand beyond binary classification to hemorrhage subtype classification, if a suitable labeled dataset is available

- [ ]  ## How to Run
   1. Download the dataset from Kaggle (link above)
   2. `pip install -r requirements.txt`
   3. Run `notebook.ipynb`
