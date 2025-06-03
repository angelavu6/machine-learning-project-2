# 🧠 Ensemble Learning & Face Recognition Classifiers

## 📁 Project Overview

This repository contains my research for the **CITS5508 Machine Learning** unit at the University of Western Australia. The research explores **ensemble learning techniques** and **dimensionality reduction** applied to two well-known datasets:

- **Part 1**: Voting Classifier on the *Diagnostic Wisconsin Breast Cancer* dataset.
- **Part 2**: Random Forests and PCA on the *Labeled Faces in the Wild* dataset.

Each part involved building machine learning pipelines, evaluating performance, and discussing trade-offs between different models and preprocessing techniques.

---

## 🧪 Technologies Used

- Python 3.11
- Jupyter Notebook
- scikit-learn
- NumPy, Pandas, Matplotlib, Seaborn

---

## 📂 Project Structure

├── assignment2.ipynb # Main Jupyter Notebook (code + analysis)

├── README.md # This file

└── /data # Folder to hold Breast Cancer dataset CSV files


---

## 🔍 Part 1 – Voting Classifier

### Dataset
- **Diagnostic Wisconsin Breast Cancer Dataset**
- Source: [UCI ML Repository](https://doi.org/10.24432/C5DW2B)

### Tasks Completed
- Preprocessed dataset: removed non-informative features
- Split into train/test (80/20) with stratification
- Trained three base models:
  - Logistic Regression
  - Random Forest
  - Support Vector Machine
- Combined them using a **soft-voting classifier**
- Evaluated each model with:
  - Precision, Recall, F1-Score
  - Confusion matrices
- Conducted performance analysis and comparison

---

## 🧑‍🎨 Part 2 – Labeled Faces in the Wild (LFW)

### Dataset
- **Labeled Faces in the Wild**
- Loaded directly via `sklearn.datasets.fetch_lfw_people`

### Tasks Completed
- Described dataset characteristics
- Sampled and visualized training images
- Trained Random Forest classifier with 1000 trees
- Evaluated model with classification report and confusion matrix
- Visualized **feature importances**
- Applied **PCA** to reduce features to 150 principal components
  - Visualized first 10 eigenfaces
  - Discussed their relevance and meaning
- Re-trained and evaluated Random Forest on PCA-transformed data
- Compared results and training time with original feature space

---

## 📊 Key Takeaways

- The Voting Classifier showed improved or stable performance compared to individual base models, highlighting the strength of ensemble methods.
- PCA significantly reduced the training time and retained high performance in face classification tasks.
- The eigenfaces analysis provided intuitive insights into how dimensionality reduction captures variance in facial features.

---

## 🧠 Skills Demonstrated

- Ensemble learning (VotingClassifier, RandomForest)
- Dimensionality reduction (PCA)
- Evaluation metrics (precision, recall, F1, confusion matrix)
- Visualization and interpretability (eigenfaces, feature importances)
- Real-world dataset handling and preprocessing

---

## 📌 How to Run

1. Download the Wisconsin Breast Cancer dataset from [UCI Repository](https://doi.org/10.24432/C5DW2B)
2. Place the data files in a `/data` folder alongside the notebook
3. Open `assignment2.ipynb` in Jupyter Notebook or Google Colab
4. Ensure you are running Python 3.11+ with the required libraries installed (`scikit-learn`, `matplotlib`, `numpy`, etc.)

---



