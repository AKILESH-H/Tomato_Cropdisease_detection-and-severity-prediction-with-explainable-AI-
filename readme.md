# Tomato Crop Disease Intelligence

An educational AI project that analyzes tomato leaf images to predict one of 10 disease classes, estimate disease severity, provide recommendations, and generate a Grad-CAM visualization when supported by the model architecture.

> **Disclaimer:** This is an AI-assisted demonstration, not a confirmed agricultural diagnosis. Grad-CAM is an explanatory visualization, not proof of causation. The displayed health score is a heuristic and is not a validated crop-health measurement.

## Features

- 10-class tomato disease classification using EfficientNetV2B0 and TensorFlow/Keras.
- Separate five-class severity prediction model.
- Grad-CAM attention visualization, when available.
- Disease-specific recommendations.
- Streamlit web interface for image upload and prediction.
- Downloadable text analysis report.
- Scripts for dataset-manifest preparation, training, and evaluation.

## Disease classes

1. Bacterial spot
2. Early blight
3. Healthy
4. Late blight
5. Leaf mold
6. Septoria leaf spot
7. Spider mites (two-spotted spider mite)
8. Target spot
9. Tomato mosaic virus
10. Tomato yellow leaf curl virus

## Tech stack

| Purpose | Technology |
|---|---|
| Language | Python |
| Deep learning | TensorFlow / Keras |
| Disease model | EfficientNetV2B0 |
| Explainability | Grad-CAM |
| Web interface | Streamlit |
| Image processing | Pillow, NumPy, TensorFlow |
| Data handling | pandas |
| Evaluation | scikit-learn |
| Visualizations | Matplotlib |

## Repository structure

```text
Tomato-Disease-Intelligence/
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── analyze_tomato.py
│   ├── prepare_10class_manifest.py
│   ├── train_disease_model_10class.py
│   └── evaluate_disease_model_10class.py
├── models/
│   ├── disease_model_10class.keras
│   ├── disease_classes_10class.json
│   └── best_severity_model_clean.keras
├── data/                    # local datasets/uploads; normally not committed
├── results/                 # generated files; normally not committed
└── PlantVillage-Dataset/    # obtain separately; do not commit by default
```

The app requires all three model artifacts listed in `models/`. `.keras` files can be large, so only commit them if their size and license terms permit it. If they are not included, document how users can obtain them. The severity model is separate and is not trained by the disease-model training script.

## Dataset

The disease classifier expects PlantVillage tomato images organized into class folders under:

```text
PlantVillage-Dataset/raw/color/
```

The dataset is not included by default. Obtain it from its original source and review its license and terms before using or redistributing it. The manifest script creates a CSV of image paths, class names, and train/validation/test splits (approximately 70%/15%/15%).

The severity model is a separate artifact expected at `models/best_severity_model_clean.keras`. This repository's listed training script does not train that model.

## Requirements

- Python compatible with your TensorFlow version
- pip
- Enough RAM and disk space for TensorFlow and image processing
- Internet for installing dependencies and obtaining datasets/weights as needed

Use a virtual environment to keep project dependencies separate from other Python projects.

## Requirements

### Software
- Python version compatible with the TensorFlow version you install.
- `pip` for installing Python packages.
- Visual Studio Code or another Python editor (optional).
- Git, if you plan to clone and version-control the project.

### Python packages
The project uses these main packages:

| Package | Purpose |
|---|---|
| `tensorflow` | Loads and runs the disease and severity models; supports training |
| `streamlit` | Runs the web interface |
| `numpy` | Numerical and image-array processing |
| `pandas` | Reads the training manifest and handles tabular data |
| `scikit-learn` | Calculates classification metrics and confusion matrix |
| `matplotlib` | Generates evaluation charts and Grad-CAM visualizations |
| `Pillow` | Opens and resizes uploaded images |

Install the packages from the project root:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow streamlit numpy pandas scikit-learn matplotlib Pillow
```

For a reproducible setup, pin and test package versions that work with your Python and TensorFlow versions. The exact compatible versions can differ by operating system and Python release.

### Data and model files
- The PlantVillage tomato image dataset is needed to regenerate the manifest and retrain the disease classifier.
- `models/disease_model_10class.keras` is required for disease predictions.
- `models/disease_classes_10class.json` maps output indices to disease labels.
- `models/best_severity_model_clean.keras` is required for severity predictions.
- The severity model is separate; the disease-training script does not train it.
- The app can only run the complete pipeline after the required model files are available in `models/`.

## Installation and execution

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/Tomato-Disease-Intelligence.git
cd Tomato-Disease-Intelligence
```

Replace `YOUR-USERNAME` with your GitHub username and use your actual repository URL.

### 2. Create and activate a virtual environment

**Windows PowerShell:**

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation in the current terminal:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

**macOS/Linux:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

The packages are listed in the **Requirements** section above. Install them with:

```bash
python -m pip install --upgrade pip
python -m pip install tensorflow streamlit numpy pandas scikit-learn matplotlib Pillow
```

### 4. Add the model files

Place these files in `models/`:

```text
disease_model_10class.keras
disease_classes_10class.json
best_severity_model_clean.keras
```

Without the required model files, the application cannot perform all predictions. Obtain the artifacts from a permitted source or train the relevant model.

### 5. Run the app

From the repository root, with the virtual environment activated:

```bash
python -m streamlit run app.py
```

Open the local URL printed in the terminal, usually `http://localhost:8501`. Upload a JPG, JPEG, or PNG tomato leaf image and click **Analyze image**. Review the predicted disease, confidence, severity, heuristic health score, recommendation, and Grad-CAM visualization if available. The report can be downloaded from the interface.

Stop the app using `Ctrl+C` in the terminal.

## Train and evaluate the 10-class disease model

Skip training if the trained disease model and class JSON already exist. Training can take considerable time and memory.

### 1. Prepare the manifest

Ensure the dataset is in the expected folder, then run:

```bash
python -m src.prepare_10class_manifest
```

This creates `results/disease_10class_manifest.csv`.

### 2. Train the disease model

```bash
python -m src.train_disease_model_10class
```

This trains an EfficientNetV2B0 classifier and saves model artifacts and training logs. It does **not** train the severity model.

### 3. Evaluate the disease model

```bash
python -m src.evaluate_disease_model_10class
```

This creates a classification report and confusion-matrix image in `results/`.

## Reported development results

During development, the 10-class disease classifier achieved a reported test accuracy of **94.81%**. The separate severity classifier had a reported test accuracy of approximately **68.41%** on its clean test split. These are development results, not a guarantee of performance on new field images. Keep the disease and severity metrics clearly distinguished and include the generated evaluation report when presenting them.

## Generated outputs

Depending on which scripts are run, `results/` may contain the dataset manifest, training log/history, classification report, confusion matrix, analysis text reports, and Grad-CAM images. Generated files and uploaded images should normally be excluded from version control unless you intentionally select a small, non-sensitive example for documentation.

## Troubleshooting

- **Model file missing:** confirm all required model files are in `models/` and filenames match the analyzer code.
- **Streamlit not found:** activate `.venv`, run `python -m pip install streamlit`, then `python -m streamlit run app.py`.
- **Manifest missing:** run `python -m src.prepare_10class_manifest` after arranging the dataset.
- **Training is slow:** close heavy applications, lower `BATCH_SIZE` if necessary, or use the existing trained model instead of retraining.
- **Grad-CAM unavailable:** some model architectures may not expose compatible layers; predictions may still work.

## Responsible use and licensing

Verify predictions with a qualified agricultural specialist before making crop-treatment decisions. Do not interpret confidence as a guarantee. Review the dataset and pretrained-weight licenses before redistribution. Add a repository `LICENSE` only after deciding how your own code may be used; a repository license does not automatically cover third-party datasets or weights.
