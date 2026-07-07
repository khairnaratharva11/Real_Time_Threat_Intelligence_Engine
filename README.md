# Real-Time Threat Intelligence & URL Classification Engine

## 📌 Overview
This repository contains an advanced machine learning pipeline designed for cybersecurity. It functions as a Real-Time Threat Intelligence Engine, dynamically extracting lexical and structural features from URLs to classify them as safe, malicious, or phishing. 

The system utilizes a **Multi-Model Heterogeneous Ensemble**. Instead of relying on a single algorithm or dataset, it trains two separate suites of models (Random Forest, Logistic Regression, and Gradient Boosting) on two distinct datasets. A custom **Soft-Voting** mechanism then aggregates the threat probabilities across all six models to deliver a final, highly robust verdict.

## 🚀 Incremental Approach & Architecture
The project is structured into three distinct scripts to separate the training phases from the real-time prediction engine:

* **`train_model_1.py` (General Threat Detection):** * Loads the `malicious_urls.csv` dataset.
  * Performs Exploratory Data Analysis (EDA) and saves distribution charts.
  * Trains three baseline models (RF, LR, GB) to recognize general malicious URL patterns.
  * Serializes and exports the models, scalers, and encoders as `.pkl` artifacts.
* **`train_model_2.py` (Phishing-Specific Detection):** * Loads a targeted `dataset_phishing.csv` dataset.
  * Trains a second, independent suite of the same three algorithms (RF, LR, GB) specifically tuned to catch phishing indicators.
  * Serializes these phishing-specific artifacts.
* **`predict_model.py` (The Real-Time Engine):** * The core application script. It accepts a target URL via command line or interactive input.
  * Dynamically parses the URL to extract a unified set of features (e.g., entropy, suspicious word counts, risky TLDs).
  * Loads all 6 pre-trained models from the previous scripts.
  * Calculates a **Global Soft Vote** by averaging the threat probabilities across the entire ensemble to output a final security verdict and confidence score.

## 🛠️ Technologies Used
* **Language:** Python 3
* **Data Processing:** Pandas, NumPy, `urllib.parse`, `re` (Regex)
* **Machine Learning:** Scikit-Learn (Random Forest, Gradient Boosting, Logistic Regression)
* **Serialization:** Joblib
* **Visualization:** Seaborn, Matplotlib

## ⚙️ How to Run
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed: `pip install pandas numpy scikit-learn matplotlib seaborn joblib`.
3. Ensure your datasets (`malicious_urls.csv` and `dataset_phishing.csv`) are in the root directory.
4. **Step 1:** Train the first set of models:
   ```bash
   python train_model_1.py
