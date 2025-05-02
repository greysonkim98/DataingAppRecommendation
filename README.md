
# 📝 Multiclass Classification of Dating App Reviews Based on User Intent

A **machine learning project** to analyze and classify user reviews from dating apps (Bumble, Tinder, Hinge), combining **clustering (K-Means)** and **classification (Softmax Regression)** to build a recommendation system based on user preferences.

## 📌 Project Overview

This project explores how real-world user reviews reflect distinct patterns across dating apps. It has two key components:

1. **Unsupervised Clustering (K-Means Algorithm)**  
   - Uses **Sentence-BERT** embeddings to group reviews into semantic clusters.
   - Extracts representative sentences from each cluster to understand key themes.
   - Visualizes clusters with **PCA** and tracks **SSE** over iterations.

2. **Supervised Classification (Softmax Regression)**  
   - Defines **7 human-centered features** based on insights from clustering.
   - Scores each review on these features using cosine similarity.
   - Trains a **manual Softmax Classifier** (without scikit-learn) to predict which app matches user preferences.

Final output: A system that recommends the most suitable dating app based on user input.

---

## 🚀 Key Features

✅ Embeds reviews using **Sentence-BERT (all-MiniLM-L6-v2)**  
✅ Clusters reviews with **K-Means** and extracts representative sentences  
✅ Scores reviews on 7 custom features (e.g., "Committed Relationship", "Willingness to Pay for Love")  
✅ Implements **manual Softmax Regression** from scratch  
✅ Includes **evaluation metrics**: accuracy, precision, recall, F1-score, confusion matrix  
✅ Interactive **user input questionnaire** to recommend the best app  

---

## 🗂️ Repository Structure

```
📁 Project Root
├── review_kmeas.py         # Clustering script using K-Means and Sentence-BERT
├── review_classification.py # Softmax classifier implementation
├── Project report.docx      # Project documentation/report
├── README.md                # This README file
├── *.csv                    # (Generated) balanced and scored review datasets
├── *.npy                    # (Generated) embedding and model parameter files
```

---

## 💾 Data Sources

Google Play Store reviews were collected from Kaggle:

- [Bumble Reviews Dataset](https://www.kaggle.com/datasets/shivkumarganesh/bumble-dating-app-google-play-store-review)
- [Tinder Reviews Dataset](https://www.kaggle.com/datasets/shivkumarganesh/tinder-google-play-store-review)
- [Hinge Reviews Dataset](https://www.kaggle.com/datasets/sidharthkriplani/datingappreviews)

Each dataset includes: `reviewId`, `userName`, `content`, `score`, `appVersion`, etc.

---

## 🛠️ How to Run

1. **Install dependencies:**

```bash
pip install pandas numpy matplotlib scikit-learn sentence-transformers
```

2. **Run Clustering (K-Means):**

```bash
python review_kmeas.py
```

Generates balanced dataset, embeddings, and extracts cluster representatives.

3. **Run Classification (Softmax Regression):**

```bash
python review_classification.py
```

Trains or loads model, evaluates, and prompts user questionnaire to recommend app.

---

## 🎯 Example User Interaction

At runtime, users are asked to rate 7 preference statements (0-10).  
The model predicts the app with the highest match probability.

```
Please rate how much you agree with each statement from 0 to 10:
1. How important is it for you to find a serious, long-term relationship? (0-10): 8
2. How comfortable are you with initiating conversations first? (0-10): 7
...
```

✅ Output:

```
Recommendation Ranking:
1. Bumble - 72.50% match
2. Hinge - 60.30% match
3. Tinder - 49.20% match
```

---

## 📊 Results

Achieved **~59% accuracy** in classifying app reviews, outperforming random guessing.  
Clustering revealed distinct themes per app (e.g., poor customer service, frustration with paid features).

---

## ✍️ Author

**Minjae Kim**  
CPSC 483 Project, California State University, Fullerton

---

## 📃 License

This project is for academic purposes.
