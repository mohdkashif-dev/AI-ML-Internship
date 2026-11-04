# AI/ML Internship - Task 1

**AI/ML Development Environment Setup, Python Foundations & Data Exploration**

Author: Kashif | Internship domain: Artificial Intelligence & Machine Learning

This repository contains the complete Task 1 submission: a configured development environment, Python fundamentals, data cleaning, exploratory data analysis (EDA), a mini project (*Student Performance Data Explorer*), the Iris advanced challenge, and initial EDA for the capstone (*AI-Based Employee Attrition Prediction System*).

## Project structure

```
AI-ML-Internship/
├── data/
│   ├── iris.csv                    # Iris dataset (150 rows)
│   ├── student_data.csv            # raw student performance data (509 rows, with gaps/duplicates)
│   ├── student_data_clean.csv      # cleaned output of notebook 02
│   ├── employee_data.csv           # raw employee data (capstone)
│   └── employee_data_clean.csv     # cleaned output of notebook 04
├── notebooks/
│   ├── 01_python_fundamentals.ipynb
│   ├── 02_student_performance_explorer.ipynb   # Parts D, E, F (mini project)
│   ├── 03_iris_eda.ipynb                       # Part G (advanced challenge)
│   └── 04_employee_attrition_eda.ipynb         # capstone progress
├── src/
│   ├── generate_datasets.py        # reproducible dataset generation
│   ├── eda_utils.py                # reusable cleaning / plotting helpers
│   ├── make_diagrams.py            # the four workflow diagrams
│   └── make_report.py              # builds the PDF report
├── reports/EDA_Report.pdf          # PDF report summarising EDA findings
├── images/                         # charts and diagrams (PNG)
├── models/                         # reserved for Task 2
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
# 1. clone and enter the project
git clone <your-repo-url> AI-ML-Internship
cd AI-ML-Internship

# 2. create and activate a virtual environment
python -m venv venv
venv\Scripts\activate            # Windows
source venv/bin/activate         # Linux / macOS

# 3. install libraries
pip install -r requirements.txt

# 4. (optional) regenerate datasets, then launch Jupyter
python src/generate_datasets.py
jupyter notebook
```

Run the notebooks in order (01 -> 04) from inside the `notebooks/` folder. To rebuild the diagrams and PDF report: `python src/make_diagrams.py` then `python src/make_report.py`.

## Datasets

| File | Description |
|------|-------------|
| `iris.csv` | Classic Iris dataset (bundled with scikit-learn) |
| `student_data.csv` | **Synthetic** student performance data (seeded, reproducible) with deliberately injected missing values and duplicates |
| `employee_data.csv` | **Synthetic** HR data for the attrition capstone (seeded) |

The student and employee datasets are synthetic because no data files were supplied with the task. To use real data (e.g. a Kaggle *Student Performance* or *IBM HR Attrition* CSV), drop it into `data/` with the same column names or adjust the column names in the notebooks.

## What each notebook covers

| Notebook | Content |
|----------|---------|
| 01 Python fundamentals | variables, types, operators, conditionals, loops, functions, lists, tuples, dicts, sets, comprehensions, file handling, exceptions |
| 02 Student explorer | load CSV, missing values, duplicates, rename columns, mean/median/mode, histograms, box plots, scatter plots, correlation heatmap, insights |
| 03 Iris EDA | full EDA incl. pair plot, species comparison and insights |
| 04 Employee attrition | data quality, attrition drivers, correlations, observations for Task 2 |

## Key findings

**Students:** study hours is the strongest driver of marks (r ≈ 0.53 with math, 0.64 with reading); completing test prep adds about 5 points in math; attendance helps modestly; sleep and gender have little effect.

**Iris:** petal length and width are almost perfectly correlated (r = 0.96) and separate the three species; sepal width is the weakest feature.

**Employees:** overall attrition 26.1%; overtime (38.6% vs 20.9%) and low job satisfaction are the clearest signals; income and tenure correlate negatively with attrition. Target is imbalanced, so stratified splitting and recall/F1 will matter in Task 2.

## Diagrams

Generated in `images/`: `diagram_ml_lifecycle.png`, `diagram_folder_structure.png`, `diagram_eda_workflow.png`, `diagram_git_workflow.png`.

## Git workflow used

```bash
git init
git add .
git commit -m "meaningful message"
git branch -M main
git remote add origin <your-repo-url>
git push -u origin main
```

## Next: Task 2

Preprocessing, feature engineering, train/test split, baseline models and evaluation on the cleaned datasets from this task.
