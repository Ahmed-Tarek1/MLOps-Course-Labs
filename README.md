# Bank Customer Churn Prediction - MLflow Project

## Overview

This project implements machine learning models to predict **bank customer churn** using historical customer data. The project is designed with **MLOps principles**: experiment tracking, model logging, and versioned deployment using **MLflow**.

You can train, evaluate, and track multiple models, including **Logistic Regression**, **Random Forest**, and **Gradient Boosting**. Preprocessing is modular and logged as an artifact.

---

## Features

- **Preprocessing:**
  - Standard scaling of numeric features
  - One-hot encoding of categorical features
  - Target rebalancing via downsampling

- **Models:**
  - Logistic Regression
  - Random Forest Classifier
  - Gradient Boosting Classifier

- **MLflow Integration:**
  - Logging parameters, metrics, and tags
  - Logging model artifacts (trained model + preprocessor)
  - Confusion matrix as artifact
  - Experiment tracking and model versioning

- **Modular codebase:**
  - `preprocessing.py` → data prep & transformer
  - `models.py` → model training
  - `train.py` → orchestrates experiments and MLflow logging

---

## Installation

```bash
# Create virtual environment
conda create -n churn_prediction python=3.12 -y
conda activate churn_prediction

# Install dependencies
pip install -r requirements.txt

## Directory Structure
```text
Experiment_Tracking/
│
├── dataset/
│   └── Churn_Modelling.csv   # Dataset file
├── src/                      
│   ├── evaluation.py         # Evaluate the model, log metrics and confusion matrix.
│   ├── models.py             # Model training functions
│   ├── preprocessing.py      # Data cleaning & transformer logic
│   └── train.py              # Orchestration & MLflow logging
├── README.md
└── requirements.txt
