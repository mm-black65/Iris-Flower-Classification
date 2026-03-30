# 🌸 Iris Flower Classification using Machine Learning

## 📌 Overview
This project focuses on building and comparing multiple machine learning models to classify iris flowers based on their physical features such as sepal length, sepal width, petal length, and petal width.

The goal is to understand how different algorithms perform on the same dataset and analyze their strengths and weaknesses.

---

## 🚀 Features
- Data visualization using Seaborn
- Model training with:
  - K-Nearest Neighbors (KNN)
  - Logistic Regression
  - Decision Tree
- Model comparison using accuracy scores
- Confusion matrix evaluation
- Hyperparameter tuning (K value optimization for KNN)
- Interactive web app using Streamlit

---

## 🧠 Problem Statement
Can we accurately classify iris flower species using machine learning models, and which model performs best?

---

## 📊 Dataset
The dataset used is the built-in Iris dataset from Scikit-learn, which contains:
- 150 samples
- 3 classes (Setosa, Versicolor, Virginica)
- 4 features per sample

---

## ⚙️ Tech Stack
- Python
- Scikit-learn
- Pandas
- NumPy
- Matplotlib & Seaborn
- Streamlit

---

## 📈 Model Performance
| Model                | Accuracy |
|---------------------|----------|
| KNN                 | ~95-100% |
| Logistic Regression | ~95-100% |
| Decision Tree       | ~90-100% |

*(Accuracy may vary slightly depending on train-test split)*

---

## 🔍 Key Insights
- KNN performs well but is sensitive to the value of K.
- Logistic Regression works efficiently due to linear separability of data.
- Decision Trees are easy to interpret but may overfit.

---

## 🖥️ Run the Project

### 1. Clone Repository
```bash
git clone https://github.com/your-username/iris-classifier.git
cd iris-classifier
