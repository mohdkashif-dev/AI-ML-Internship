"""Generate the datasets used in Task 1.

- iris.csv          : the classic Iris dataset (from scikit-learn's bundled copy)
- student_data.csv  : synthetic student performance data (with a few injected
                      missing values and duplicate rows so cleaning is meaningful)
- employee_data.csv : synthetic HR data for the capstone (Employee Attrition)

Run from the project root:  python src/generate_datasets.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)
rng = np.random.default_rng(42)


def make_iris() -> None:
    iris = load_iris(as_frame=True)
    df = iris.frame.copy()
    df["species"] = df["target"].map(dict(enumerate(iris.target_names)))
    df = df.drop(columns="target")
    df.columns = ["sepal length (cm)", "sepal width (cm)",
                  "petal length (cm)", "petal width (cm)", "species"]
    df.to_csv(DATA_DIR / "iris.csv", index=False)


def make_students(n: int = 500) -> None:
    study = np.clip(rng.normal(3.0, 1.2, n), 0, 8).round(1)
    attendance = np.clip(rng.normal(82, 11, n), 40, 100).round(1)
    sleep = np.clip(rng.normal(7, 1.1, n), 3.5, 10).round(1)
    parent_edu = rng.choice(["High School", "Bachelors", "Masters", "PhD"],
                            n, p=[0.35, 0.4, 0.18, 0.07])
    prep = rng.choice(["Completed", "Not Completed"], n, p=[0.4, 0.6])
    gender = rng.choice(["Male", "Female"], n)
    base = 28 + 6.0 * study + 0.25 * attendance + 1.2 * sleep
    base += np.where(prep == "Completed", 5, 0)
    base += pd.Series(parent_edu).map({"High School": 0, "Bachelors": 1.5,
                                       "Masters": 3, "PhD": 4}).values
    math = np.clip(base + rng.normal(0, 7, n), 0, 100).round(0)
    reading = np.clip(base + rng.normal(2, 7, n), 0, 100).round(0)
    writing = np.clip(0.6 * reading + 0.4 * base + rng.normal(0, 5, n), 0, 100).round(0)
    # add a few unusually low/high scores so box plots show outliers
    math[rng.choice(n, 5, replace=False)] = [5, 8, 12, 100, 99]
    df = pd.DataFrame({
        "Student ID": [f"S{1000 + i}" for i in range(n)],
        "Gender": gender,
        "Parental Education": parent_edu,
        "Test Prep": prep,
        "Study Hours": study,
        "Attendance %": attendance,
        "Sleep Hours": sleep,
        "Math Score": math,
        "Reading Score": reading,
        "Writing Score": writing,
    })
    # inject missing values
    for col, k in [("Math Score", 12), ("Study Hours", 8),
                   ("Attendance %", 10), ("Parental Education", 6)]:
        df.loc[rng.choice(n, k, replace=False), col] = np.nan
    # inject duplicates
    df = pd.concat([df, df.sample(9, random_state=1)], ignore_index=True)
    df.to_csv(DATA_DIR / "student_data.csv", index=False)


def make_employees(n: int = 1200) -> None:
    age = np.clip(rng.normal(37, 9, n), 20, 60).round(0).astype(int)
    years = np.clip((age - 21) * rng.uniform(0.05, 0.6, n), 0, 35).round(0).astype(int)
    dept = rng.choice(["Sales", "R&D", "HR", "Finance", "IT"], n,
                      p=[0.3, 0.32, 0.08, 0.1, 0.2])
    role_level = np.clip(1 + years // 6 + rng.integers(0, 2, n), 1, 5)
    income = (2500 + role_level * 2600 + years * 140
              + rng.normal(0, 900, n)).round(-1).astype(int)
    overtime = rng.choice(["Yes", "No"], n, p=[0.3, 0.7])
    satisfaction = rng.integers(1, 5, n)
    wlb = rng.integers(1, 5, n)
    dist = rng.integers(1, 30, n)
    logit = (-1.6 + 1.1 * (overtime == "Yes") - 0.45 * (satisfaction - 2.5)
             - 0.00018 * (income - 9000) + 0.02 * (dist - 10)
             - 0.03 * (years) - 0.25 * (wlb - 2.5))
    p = 1 / (1 + np.exp(-logit))
    attrition = np.where(rng.random(n) < p, "Yes", "No")
    df = pd.DataFrame({
        "EmployeeID": np.arange(1, n + 1),
        "Age": age,
        "Department": dept,
        "JobLevel": role_level,
        "MonthlyIncome": income,
        "YearsAtCompany": years,
        "DistanceFromHome": dist,
        "OverTime": overtime,
        "JobSatisfaction": satisfaction,
        "WorkLifeBalance": wlb,
        "Attrition": attrition,
    })
    df.loc[rng.choice(n, 15, replace=False), "MonthlyIncome"] = np.nan
    df.loc[rng.choice(n, 10, replace=False), "JobSatisfaction"] = np.nan
    df.to_csv(DATA_DIR / "employee_data.csv", index=False)


if __name__ == "__main__":
    make_iris()
    make_students()
    make_employees()
    print("Datasets written to", DATA_DIR)
