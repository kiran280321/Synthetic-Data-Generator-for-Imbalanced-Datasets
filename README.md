
# Synthetic Data Generator for Imbalanced Datasets

## 📌 Overview
This project focuses on handling class imbalance in machine learning datasets using synthetic data generation techniques. In addition to implementing standard SMOTE, a custom cluster-based SMOTE method is developed to generate higher-quality minority samples by preserving local data structure. The solution is fully integrated with scikit-learn pipelines and evaluated using appropriate performance metrics.

---

## 🎯 Problem Statement
Imbalanced datasets often cause machine learning models to be biased toward the majority class, resulting in poor minority-class performance. Traditional accuracy metrics become misleading in such scenarios. This project addresses the issue by applying data-level resampling techniques and evaluating their impact on classifier performance.

---

## 🚀 Features
- Implementation of **standard SMOTE**
- Custom **Cluster-Based SMOTE** to reduce noisy synthetic samples
- Integration with **scikit-learn pipelines**
- Evaluation using **Precision, Recall, F1-score (minority class), and ROC-AUC**
- Visual comparison of model performance before and after resampling

---

## 🛠️ Technologies Used
- Python 3
- scikit-learn
- imbalanced-learn
- Pandas
- NumPy
- Matplotlib
- Jupyter Notebook

---

## 📁 Project Structure
```

synthetic-data-generator/
│
├── generators/
│   ├── **init**.py
│   ├── smote_generator.py
│   └── cluster_smote.py
│
├── evaluation/
│   ├── **init**.py
│   └── evaluate.py
│
├── notebook/
│   └── demo_pipeline.ipynb
│
├── requirements.txt
└── README.md

````

---

## ▶️ How to Run

### 1. Clone the Repository
```bash
git clone https://github.com/<your-username>/synthetic-data-generator.git
cd synthetic-data-generator
````

### 2. Create Virtual Environment (Optional but Recommended)

```bash
python -m venv venv
venv\Scripts\activate    # Windows
# source venv/bin/activate  # macOS/Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the Notebook

```bash
jupyter notebook
```

Open:

```
notebook/demo_pipeline.ipynb
```

Run all cells to see results and visualizations.

---

## 📊 Evaluation Metrics

Model performance is evaluated using:

* Precision (Minority Class)
* Recall (Minority Class)
* F1-Score (Minority Class)
* ROC-AUC

These metrics provide a more reliable assessment than accuracy for imbalanced datasets.

---

## 📈 Results Summary

* Baseline model performs poorly on the minority class
* Standard SMOTE significantly improves recall and F1-score
* Cluster-Based SMOTE achieves the best overall performance, with improved minority-class recall, F1-score, and ROC-AUC


## 📌 Conclusion

This project demonstrates that thoughtful data-level techniques, combined with proper evaluation metrics, can substantially improve model performance on imbalanced datasets. The modular design allows easy extension to other datasets and classifiers.

---

## 👤 Author

**V R N SUDHA KIRAN YARRAMSETTY**
B.Tech CSE, SRM University-AP



