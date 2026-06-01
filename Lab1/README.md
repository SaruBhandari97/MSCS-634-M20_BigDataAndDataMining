# MSCS 634 Lab 1 — Data Visualization, Preprocessing & Statistical Analysis

## Purpose
This lab applies data visualization, preprocessing, and statistical analysis 
techniques to the Superstore Sales dataset using Python and Jupyter Notebook.

## Dataset
- **Source:** Kaggle — Superstore Sales Dataset
- **Records:** 9,800 rows × 18 columns
- **Key variable:** Sales (transaction dollar value)

## Key Insights
- Technology is the highest-revenue product category
- Sales data is strongly right-skewed (mean $230.77, median $54.49)
- IQR outlier detection removed 1,145 transactions (11.7% of dataset)
- The West region contributes the largest share of total revenue
- No meaningful correlation between Row ID and Sales (r = 0.001)

## Files
- `Lab1.ipynb` — Jupyter Notebook with all code and outputs
- `superstore_final_dataset.csv` — Dataset used for analysis
- `Lab1_Report_APA.docx` — Full lab report in APA format

## Tools Used
- Python 3.11
- Pandas, NumPy, Matplotlib, Seaborn, Scikit-learn
- Jupyter Notebook in VS Code