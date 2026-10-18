# FaceNet Face Verification

A face-verification application built with Python, PyTorch, FaceNet, MTCNN, and Streamlit. It compares two uploaded images and estimates whether they show the same person using facial embeddings and cosine similarity.

The project also includes an evaluation pipeline using the Labeled Faces in the Wild (LFW) dataset to measure model performance on face pairs.

## Features

- Upload two images through a Streamlit interface
- Detect and align faces using MTCNN
- Generate 512-dimensional face embeddings
- Compare embeddings using cosine similarity
- Evaluate face pairs using a development-set threshold
- Calculate accuracy, precision, recall, F1-score, FAR, FRR, and ROC-AUC
- Generate ROC curves and confusion matrices
- Save test predictions, metrics, and skipped-pair records

## Technologies

- Python 3.12
- PyTorch
- facenet-pytorch
- InceptionResnetV1 pretrained on VGGFace2
- MTCNN
- scikit-learn
- NumPy
- Matplotlib
- Streamlit
- Pillow

## Project Structure

    facenet-face-verification/
    ├── evaluation/
    │   ├── __init__.py
    │   ├── dataset.py
    │   ├── metrics.py
    │   ├── run_evaluation.py
    │   ├── inspect_pair.py
    │   └── test_detection.py
    ├── results/
    │   ├── baseline_resize1/
    │   │   ├── confusion_matrix.png
    │   │   ├── metrics.json
    │   │   ├── roc_curve.png
    │   │   └── skipped_pairs.csv
    │   ├── metrics.json
    │   ├── metrics.csv
    │   ├── test_predictions.csv
    │   ├── skipped_pairs.csv
    │   ├── roc_curve.png
    │   └── confusion_matrix.png
    ├── src/
    │   ├── app.py
    │   └── facenet_model.py
    ├── .gitignore
    ├── requirements.txt
    └── README.md

## Requirements

- Python 3.12
- pip

## Installation

Clone the repository and enter the project directory:

    git clone <YOUR_GITHUB_REPOSITORY_URL>
    cd facenet-face-verification

Create a virtual environment:

    python -m venv .venv312

Activate it in Windows PowerShell:

    .\.venv312\Scripts\Activate.ps1

Install dependencies:

python -m pip install -r requirements.txt

The requirements file pins the main package versions used during development. PyTorch installation can vary by operating system and hardware. For a CPU-only setup, follow the official PyTorch installation instructions if pip cannot resolve a compatible build.

## Run the application

From the project root, run:

    python -m streamlit run src/app.py

The application opens in your browser. Upload two images containing faces to compare their embeddings.

## Run the evaluation pipeline

From the project root, run:

    python -m evaluation.run_evaluation

The evaluation pipeline uses LFW face pairs and separates threshold selection from final testing. The development subset is used to select the decision threshold, while the test subset is used to report final metrics.

The LFW dataset may need to be downloaded the first time the evaluation is run. An internet connection may be required for this initial download.

Evaluation outputs include metrics, test predictions, skipped-pair records, a ROC curve, and a confusion matrix. Exact output files depend on the evaluation script configuration.

## Evaluation Results

The current recorded test evaluation produced the following results:

| Metric | Result |
|---|---:|
| Test pairs evaluated | 966 |
| Test pairs available | 1,000 |
| Test coverage | 96.60% |
| Accuracy | 98.24% |
| Precision | 98.16% |
| Recall | 98.36% |
| F1-score | 98.26% |
| ROC-AUC | 0.9969 |
| False Acceptance Rate (FAR) | 1.88% |
| False Rejection Rate (FRR) | 1.64% |
| Approximate Equal Error Rate (EER) | 1.86% |
| Decision threshold | 0.4172 |

The threshold was selected using the development subset, not the test subset. The test results are based on the 966 pairs successfully processed; 34 of the 1,000 test pairs were skipped because at least one image could not be processed successfully.

These are the current recorded results for this configuration. Re-run the pipeline to reproduce them in your own environment.

## Understanding the Results

- **Accuracy:** Proportion of evaluated pairs classified correctly.
- **Precision:** Proportion of predicted same-person pairs that were actually same-person pairs.
- **Recall:** Proportion of actual same-person pairs correctly identified.
- **ROC-AUC:** Measures how well similarity scores distinguish same-person and different-person pairs across thresholds.
- **FAR:** Proportion of different-person pairs incorrectly accepted as a match.
- **FRR:** Proportion of same-person pairs incorrectly rejected.
- **EER:** Approximate error rate at the point where FAR and FRR are similar.

## Limitations

- The face-recognition network uses pretrained weights; it was not trained from scratch for this project.
- LFW is a benchmark dataset and does not represent every real-world environment or population.
- Some low-resolution, cropped, or difficult images may fail face detection and are excluded from the reported test metrics.
- Reported performance applies only to the evaluated pairs and configuration. It should not be assumed to represent performance on arbitrary uploaded images.
- This project is intended for learning and research, not security-critical identity verification or decisions about access, eligibility, or individuals.

## Future Improvements

- Add automated tests and configuration files for repeatable experiments.
- Compare different image preprocessing and threshold-selection strategies.
- Evaluate additional datasets and report performance across relevant subgroups.
- Add experiment tracking and more detailed error analysis.

## License and Dataset

Check the licensing terms of the model weights, dependencies, and LFW dataset before redistributing them or using this project commercially.
