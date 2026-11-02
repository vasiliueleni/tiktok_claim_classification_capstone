# TikTok Video Classification: Claim vs. Opinion

## 📌 Project Overview
This project is part of a machine learning capstone analyzing user-reported content for TikTok. The objective is to build a robust binary classification model that automatically predicts whether a video presents a **"claim"** or an **"opinion"**. Automating this initial triage helps TikTok reduce its moderation backlog and allows human review teams to prioritize potentially harmful claims more efficiently.

---

## 🛠️ Tech Stack & Libraries
* **Language:** Python
* **Data Manipulation & Analysis:** Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Machine Learning & Modeling:** Scikit-Learn (Random Forest, GridSearchCV, train_test_split), XGBoost
* **Model Serialization:** Joblib / Pickle

---

## 📂 Project Structure
```text
├── models/                  # Saved pickled GridSearchCV model objects
├── tiktok_dataset.csv       # Raw TikTok dataset (19,382 rows)
├── Capstone_TikTok.ipynb    # Main Jupyter Notebook containing end-to-end workflow
└── README.md                # Project documentation
