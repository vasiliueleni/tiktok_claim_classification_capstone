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
```
---

## 🔬 Methodology (PACE Framework)
This project follows the PACE (Plan, Analyze, Construct, Execute) framework:
### Plan & Analyze: 
* **Ethical Considerations:** Evaluated the impact of model errors. False Negatives (missing a claim) were identified as the critical error since harmful misinformation could bypass review, making **Recall** the primary optimization metric.
* **Data Exploration:** Inspected `19,084` raw records, handled missing values (`298` rows dropped, leaving `19,084` clean rows), confirmed zero duplicates, and verified a balanced target distribution (`50.35%` claims vs. `49.65%` opinions).

### Construct: 
* **Feature Engineering:** Extracted character lengths (`text_length`) from video transcriptions (revealing claims average `95.38` characters vs. `82.72` for opinions).
* **Preprocessing:** Dummy-encoded categorical features (`verified_status`, `author_ban_status`) and partitioned the dataset into a 60/20/20 train/validation/test split (`11,450` / `3,817` / `3,817` rows)

### Execute:
* Trained and tuned both Random Forest and XGBoost classifiers via 5-fold cross-validaation (`GridSearchCV` refitting on recall)
* Selected the Random Forest model as the champion model after it slightly outperformed XGBoost on validation data.

--- 

## 📊 Key Results & Champion Model Performance
When evaluated on the unseen test set ($n = 3,817$ videos), the champion Random Forest model generalized exceptionally well:
* Precision: `1.0000` (`100.0%`) — Zero False Positives (no opinion videos were misclassified as claims).
* Recall: `0.9912` (`99.12%`) — Successfully identified over 99% of all actual claim videos.
* F1-Score: `0.9956`
* Accuracy: `0.9955` (`99.55%`) — Correctly classified `3,800` out of `3,817` total test videos, making only `17` total errors (all false negatives).

---

## Feature Importance Insights
* Feature importance analysis showed that predictions are driven almost entirely by user engagement metrics (`views`, `likes`, `shares`, `downloads`, `comments`), which account for over `99.9%` of the model's total predictive weight.

* Because opinion videos in the dataset never exceed `9,998 views`, the model successfully isolated claims using sharp threshold splits on engagement volume.

---

## 🚀 Limitations & Future Recommendations
* **Operational Limitation:** Because engagement metrics accumulate after publication, the current model is best suited for triaging existing backlogs rather than flagging brand-new uploads instantly.

* **Phase 2 Next Steps:**
  1. **Incorporate NLP Features:** Extract bag-of-words n-grams (`CountVectorizer`) or TF-IDF tokens from `video_transcription_text` to detect linguistic assertion patterns independent of view counts.
  2. **Early-Velocity Metrics:** Collect creator history and initial view/report velocity within the first hour of posting to flag newly uploaded content proactively.

👤 Author

Eleni Vasiliu

*Aspiring Data Analyst | Machine Learning Enthusiast*

[LinkedIn Profile](https://www.linkedin.com/in/elenivasiliu/)
