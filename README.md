# Auspify Technologies — Machine Learning Internship Portfolio

Completed 4 end-to-end Machine Learning projects on the Netflix catalog dataset (8,790 titles):

1. **Task 1 — Netflix Content Recommendation System (`Task_1_Recommendation.../`)**
   - Built a content-based recommendation engine using text normalization, TF-IDF vectorization (`ngram_range=(1, 2)`), and pairwise cosine similarity.
2. **Task 2 — Content Type Prediction Model (`Task_2_Content_Typ.../`)**
   - Built a binary classifier (`Movie` vs. `TV Show`) using class-weighted Random Forest (`77.36%` accuracy, `0.7321` 5-fold CV Macro F1), a standalone inference script (`predict.py`), and an interactive Streamlit web app (`app.py`).
3. **Task 3 — Netflix Audience Rating Classification (`Task_3_Rating_Classification/`)**
   - Built an 11-class audience rating classifier tuned via `RandomizedSearchCV` (`55.18%` test accuracy, `0.5422` 5-fold CV Weighted F1) deployed with an interactive Streamlit UI (`app.py`).
4. **Task 4 — Netflix Content Segmentation (`Task_4_Content_Segmentation/`)**
   - Segmented the catalog into 5 distinct content clusters using `StandardScaler`, K-Means clustering ($K=5$), and 2D PCA visualization.