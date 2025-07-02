# 📊 Churn Prediction Model – DeepQ-AI Assignment 1

This repository contains the solution for Assignment 1 of the AI Engineer Internship process at **DeepQ-AI**. The goal is to build a classification model that predicts whether a customer will churn based on various features provided in a dataset.

---

## 🚀 Objective

- Build a **classification model** to predict the binary target column: `Target_ChurnFlag`.
- Derive insights from the dataset and share key findings in a presentation.
- **Deploy the model** with a user-friendly interface using **Streamlit**.

---

## 📁 Dataset Overview

- **Total Rows:** 167,000+
- **Features:** 
  - `UID`: Unique identifier
  - `X0` to `X215`: Independent features (numerical, categorical, date formats)
  - `Target_ChurnFlag`: Binary target column indicating churn

---

## 🧹 Preprocessing Steps

- Dropped columns with >50% missing values.
- Removed identifier (`UID`) and leaked feature (`X16`).
- Handled date columns by parsing and extracting year/month/day.
- Label-encoded categorical variables.
- Imputed missing values with the median.
- Addressed class imbalance using **SMOTE**.

---

## 🧠 Model Building

- Used **Random Forest Classifier** with `class_weight='balanced'`.
- Performed a **stratified train-test split**.
- Tuned model using validation metrics.

---

## 📈 Model Performance

| Metric          | Class 0 | Class 1 | Overall |
|-----------------|---------|---------|---------|
| Precision       | 0.91    | 0.88    |         |
| Recall          | 0.91    | 0.88    |         |
| F1-score        | 0.91    | 0.88    |         |
| Accuracy        |         |         | **0.90** |
| ROC-AUC Score   |         |         | **0.946** |

> The model demonstrates excellent performance and balance across both classes, thanks to SMOTE and careful feature handling.

---

## 🌐 Deployment

The model has been deployed using **Streamlit**, allowing users to:
- Upload custom feature inputs of csv/.xlxs file.
- Receive churn predictions in real-time.

🔗 **Live Demo:** https://churn-preditor-app.streamlit.app/

---

## 🖥️ Clone and Run Locally

Follow these steps to set up the project on your local machine:

### ✅ Prerequisites

- Python 3.7+
- pip
- Virtual environment (recommended)

### 📦 Installation

```bash
# Step 1: Clone the repository
git clone https://github.com/your-username/deepq-churn-prediction.git
cd deepq-churn-prediction

# Step 2: Create and activate virtual environment (optional but recommended)
python -m venv venv
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Step 3: Install dependencies
pip install -r requirements.txt

#Step 4: 🚀 Run Streamlit App
streamlit run streamlit_app.py

## 🗂 Files Included

- `model_training.ipynb` – End-to-end code for preprocessing, model training, and evaluation.
- `churn_model.pkl` – Saved model on drive for deployment(Download from GoogleDrive).
- `streamlit_app.py` – Streamlit script to serve predictions.
- `Churn_Prediction_Presentation.pdf` – Summary of insights, process, and evaluation.
- `README.md` – Project overview and documentation.

---

## 🧑‍💻 Tools & Libraries

- Python
- Pandas, NumPy, Scikit-learn
- Imbalanced-learn (SMOTE)
- Streamlit

---

## 📬 Contact

**Author:** Sayan Maity  
**Email:** sayanmaity2003@gmail.com 
**LinkedIn:** [[Optional links here]](https://www.linkedin.com/in/sayan-maity-718974253)

---

Thank you for reviewing this project. I look forward to your feedback!

