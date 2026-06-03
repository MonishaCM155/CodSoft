# CodSoft Data Science Internship Tasks

> **Intern:** Monisha C M  
> **Internship:** Data Science — CodSoft  
> **Tasks Completed:** 4 / 5

---

## Task 1: Titanic Survival Prediction
**Goal:** Predict whether a passenger survived the Titanic disaster.  
**Dataset:** Titanic-Dataset.csv (891 rows, 12 features)  
**Models:** Logistic Regression, Random Forest  
**Best Result:** Random Forest — **Accuracy: 81.56%**  
**Key Features:** Sex, Fare, Age, Pclass, FamilySize

---

## Task 2: Movie Rating Prediction
**Goal:** Predict movie ratings based on genre, director, actors, votes, and year.  
**Dataset:** IMDb Movies Dataset (7,668 movies)  
**Models:** Linear Regression, Random Forest, Gradient Boosting  
**Best Result:** Gradient Boosting — **R² = 0.80, RMSE = 0.447**  
**Key Features:** votes, director_avg_score, star_avg_score, year

---

## Task 3: Iris Flower Classification
**Goal:** Classify Iris flowers into Setosa, Versicolor, Virginica.  
**Dataset:** IRIS.csv (150 samples, 4 features)  
**Models:** KNN, Decision Tree, Random Forest, SVM  
**Best Result:** All models — **Accuracy: 100%**  
**Key Features:** petal_length, petal_width (most discriminative)

---

## Task 4: Sales Prediction Using Python
**Goal:** Forecast product sales from advertising spend (TV, Radio, Newspaper).  
**Dataset:** Advertising Dataset (200 samples)  
**Models:** Linear Regression, Random Forest  
**Best Result:** Linear Regression — **R² = 0.935, RMSE = 1.246**  
**Key Insight:** TV and Radio spend are the strongest predictors; Newspaper has minimal impact.

---

## Project Structure
```
CODSOFT/
├── task1_titanic.py          # Titanic survival prediction
├── task2_movie_rating.py     # Movie rating prediction
├── task3_iris.py             # Iris flower classification
├── task4_sales_prediction.py # Sales prediction
├── task1_eda.png             # EDA visualizations - Titanic
├── task1_results.png         # Model results - Titanic
├── task2_eda.png             # EDA visualizations - Movies
├── task2_results.png         # Model results - Movies
├── task3_eda.png             # EDA visualizations - Iris
├── task3_results.png         # Model results - Iris
├── task4_eda.png             # EDA visualizations - Sales
├── task4_results.png         # Model results - Sales
└── README.md
```

## Libraries Used
- `pandas`, `numpy` — data manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — machine learning models

## How to Run
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
python task1_titanic.py
python task2_movie_rating.py
python task3_iris.py
python task4_sales_prediction.py
```
