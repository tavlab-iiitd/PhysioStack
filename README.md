## 📈 PhysioStack: Resolution-Adaptive Models for Physiological Time Series

PhysioStack is a family of BERT-style models trained on symbolic representations of ICU vital signs at multiple temporal resolutions (e.g., 5, 10, 15, 30, 60 minutes).  
This repository contains the whole pipeline:

- Data preprocessing and cleaning  
- Symbolic conversion of vital-sign time series  
- Hyperparameter optimization with Optuna  
- Masked Language Modeling (MLM) training of PhysioStack  
- Evaluation (MLM accuracy, resolution transfer)  
- Scripts to reproduce key figures and results

---

## 🔍 1. Overview

Modern ICUs generate rich, high-frequency physiological time series. PhysioStack learns resolution-adaptive representations of these signals to support:

- Imputation and forecasting
- Early risk stratification
- Downstream models (e.g., shock prediction)

This codebase is designed to be:

- **Modular** – separate modules for data, symbols, training, evaluation
- **Config-driven** – experiments use JSON configs for model and tokenizer settings
- **Reproducible** – best parameters and configs are saved for repeatable results

---

## 🗂️ 2. Repository Structure

- **PhysioStack/**  
  - `scripts/` – command-line tools  
  - `notebooks/` – exploration and visualization   
  - `results/` – generated plots and evaluation metrics   
- **README.md**  
- **.gitignore**

---

## 📊 3. Datasets

PhysioStack is trained and evaluated on three ICU datasets.  
Links to all datasets are provided below.  
Please note that MIMIC and eICU require authorized access through PhysioNet, while SAFE-ICU requires authorized access through the SafeICU database.



### 🔹 3.1 SAFE-ICU (AIIMS Delhi, IIIT-Delhi)
High-resolution pediatric ICU dataset containing vital signs, treatment charts, labs, and notes.

📌 **Dataset Link:**  
🔗 *https://safeicu.aiims.edu.in/*



### 🔹 3.2 MIMIC-III / MIMIC-IV (PhysioNet)
Large adult ICU datasets containing physiological time series, labs, interventions, and clinical documentation.  
Access requires PhysioNet credentialing.

📌 **Official PhysioNet Pages:**  
- MIMIC-III: https://physionet.org/content/mimiciii/1.4/  



### 🔹 3.3 eICU Collaborative Research Database (PhysioNet)
Multi-center adult ICU dataset from over 200 hospitals across the USA.  
This dataset also requires PhysioNet-approved access.

📌 **Official PhysioNet Page:**  
- EICU: https://physionet.org/content/eicu-crd/2.0/



> **Note:** This repository does not include any raw patient data.  
> Users must download datasets directly from their official sources according to their licensing requirements.

---

## 🧠 4. Models

Pretrained PhysioStack models (across multiple temporal resolutions) are packaged inside a Docker image for easy access and reproducibility.

### 🔹 4.1 Pull the Docker Image

`docker pull falconnew/tempovital_rt:v1`

### 🔹 4.2 Run the Container

`docker run -it --gpus all falconnew/tempovital_rt:v1 bash`

> **Note:** GPU support is optional but recommended for running inference or training additional models. 

### 🔹 4.3 Locate the Pretrained Models

After entering the container, the pretrained PhysioStack models are available at:

`ls /opt/models`

> **Note:** This directory contains all trained model folders, including configuration files (config.json, tokenizer.json, vocab.txt) and model weights (pytorch_model.bin).
> These models correspond to multiple temporal resolutions (5, 10, 15, 30, 60 minutes).
> You can directly load these models into your Python workflows or use them for downstream tasks (e.g., shock prediction, feature extraction).

### 🔹 4.4 Loading the Models in Python

An example Jupyter notebook demonstrating how to load and use the pretrained PhysioStack models is provided in the repository:

**`PhysioStack/notebooks/load_models.ipynb`**

---

## 📄 License

This project is licensed under the MIT License.

Please refer to the [LICENSE](LICENSE) file for terms.

---

## 👥 Contributors

- TavLab, IIIT-Delhi – Research, Design, and Development
- Open for collaboration and contributions!

---

