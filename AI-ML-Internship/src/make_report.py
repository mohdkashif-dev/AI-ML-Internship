"""Build reports/EDA_Report.pdf from the cleaned datasets and saved charts.

Run from the project root:  python src/make_report.py
"""
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, PageBreak, Paragraph, SimpleDocTemplate,
                                Spacer, Table, TableStyle)

ROOT = Path(__file__).resolve().parent.parent
IMG, DATA = ROOT / "images", ROOT / "data"

ss = getSampleStyleSheet()
H1 = ParagraphStyle("H1", parent=ss["Heading1"], fontSize=18, spaceAfter=8, textColor=colors.HexColor("#1F4E79"))
H2 = ParagraphStyle("H2", parent=ss["Heading2"], fontSize=13, spaceBefore=10, spaceAfter=4, textColor=colors.HexColor("#1F4E79"))
BODY = ParagraphStyle("B", parent=ss["BodyText"], fontSize=10, leading=14)
BUL = ParagraphStyle("BUL", parent=BODY, leftIndent=14, bulletIndent=4)
CAP = ParagraphStyle("CAP", parent=BODY, fontSize=8.5, textColor=colors.grey, alignment=1)


def img(name, width=16 * cm):
    from PIL import Image as PILImage
    w, h = PILImage.open(IMG / name).size
    return [Image(str(IMG / name), width=width, height=width * h / w),
            Paragraph(name.replace("_", " ").replace(".png", ""), CAP), Spacer(1, 6)]


def bullets(items):
    return [Paragraph(t, BUL, bulletText="•") for t in items]


def table(df, col_widths=None):
    data = [list(df.columns)] + df.values.tolist()
    t = Table(data, colWidths=col_widths, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1F4E79")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 8.5),
        ("GRID", (0, 0), (-1, -1), 0.4, colors.lightgrey),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F4F6F8")]),
    ]))
    return t


def build():
    raw = pd.read_csv(DATA / "student_data.csv")
    stu = pd.read_csv(DATA / "student_data_clean.csv")
    iris = pd.read_csv(DATA / "iris.csv")
    emp = pd.read_csv(DATA / "employee_data_clean.csv")

    r = stu.select_dtypes("number").corr()
    prep = stu.groupby("test_prep")["math_score"].mean()
    sc = ["math_score", "reading_score", "writing_score"]
    summ = stu[sc].agg(["mean", "median", "std"]).round(2).T.reset_index()
    summ.columns = ["Score", "Mean", "Median", "Std Dev"]

    ir = iris.drop(columns="species").corr()
    iris_means = iris.groupby("species").mean().round(2).reset_index()
    iris_means.columns = [c.replace(" (cm)", "") for c in iris_means.columns]

    ot = emp.groupby("OverTime")["attrition_flag"].mean() * 100
    sat = emp.groupby("JobSatisfaction")["attrition_flag"].mean() * 100
    ec = emp.select_dtypes("number").corr()["attrition_flag"]

    s = []
    s += [Paragraph("AI/ML Internship - Task 1", H1),
          Paragraph("Development Environment Setup, Python Foundations &amp; Exploratory Data Analysis", H2),
          Paragraph("Report summarising the EDA findings for the Student Performance Data Explorer, "
                    "the Iris dataset challenge, and the initial Employee Attrition analysis.", BODY),
          Spacer(1, 8)]

    s += [Paragraph("1. Environment and workflow", H2)]
    s += bullets(["Python 3, VS Code, Git and Jupyter Notebook with an isolated virtual environment (<font face='Courier'>python -m venv venv</font>).",
                  "Libraries: NumPy, Pandas, Matplotlib, Seaborn (see <font face='Courier'>requirements.txt</font>).",
                  "Project organised with data / notebooks / src / reports / images / models folders and version-controlled with Git."])
    s += img("diagram_ml_lifecycle.png", 15.5 * cm)
    s += img("diagram_folder_structure.png", 11 * cm)
    s += img("diagram_eda_workflow.png", 16 * cm)
    s += img("diagram_git_workflow.png", 15.5 * cm)

    s += [PageBreak(), Paragraph("2. Student Performance Data Explorer", H1)]
    s += [Paragraph(f"<b>Dataset:</b> {raw.shape[0]} rows x {raw.shape[1]} columns (synthetic, reproducible via "
                    f"<font face='Courier'>src/generate_datasets.py</font>). <b>Data quality:</b> "
                    f"{int(raw.duplicated().sum())} duplicate rows and {int(raw.isnull().sum().sum())} missing cells were found. "
                    f"Duplicates were removed; numeric gaps filled with the median and categorical gaps with the mode. "
                    f"Cleaned shape: {stu.shape[0]} x {stu.shape[1]}.", BODY), Spacer(1, 6),
          table(summ, [4 * cm, 3 * cm, 3 * cm, 3 * cm]), Spacer(1, 8)]
    s += img("student_missing_values.png", 12 * cm)
    s += img("student_histograms.png")
    s += img("student_boxplots.png", 15 * cm)
    s += img("student_scatter.png", 15 * cm)
    s += img("student_correlation_heatmap.png", 11 * cm)
    s += img("student_group_bars.png", 15 * cm)
    s += [Paragraph("Key findings", H2)]
    s += bullets([
        f"Study hours is the strongest driver of marks (correlation with math {r.loc['study_hours', 'math_score']:.2f}, reading {r.loc['study_hours', 'reading_score']:.2f}).",
        f"Reading and writing scores are tightly linked (r = {r.loc['reading_score', 'writing_score']:.2f}).",
        f"Attendance has a smaller positive effect (r = {r.loc['attendance', 'math_score']:.2f} with math); sleep hours are weak (r = {r.loc['sleep_hours', 'math_score']:.2f}).",
        f"Students who completed test prep average {prep['Completed']:.1f} in math vs {prep['Not Completed']:.1f} without it.",
        "Box plots show a handful of extreme math scores (both very low and perfect), flagged by the 1.5 x IQR rule.",
        "The gender gap in average scores is negligible.",
    ])

    s += [PageBreak(), Paragraph("3. Iris dataset - advanced challenge", H1)]
    s += [Paragraph(f"150 observations, 4 numeric features and 3 balanced species. No missing values; {int(iris.duplicated().sum())} duplicate row removed. Mean measurements per species:", BODY), Spacer(1, 4),
          table(iris_means), Spacer(1, 8)]
    s += img("iris_histograms.png", 13 * cm)
    s += img("iris_boxplots.png", 13 * cm)
    s += img("iris_scatter.png", 15 * cm)
    s += img("iris_correlation_heatmap.png", 9 * cm)
    s += [Paragraph("Key findings", H2)]
    s += bullets([
        f"Petal length and petal width are almost perfectly correlated (r = {ir.loc['petal length (cm)', 'petal width (cm)']:.2f}).",
        "Setosa is clearly separable on petal size; versicolor and virginica overlap in sepal measurements but separate well on petals.",
        f"Sepal width is the weakest feature (r = {ir.loc['sepal width (cm)', 'petal length (cm)']:.2f} with petal length).",
        "Petal measurements are the best candidate features for a classifier in Task 2.",
    ])

    s += [PageBreak(), Paragraph("4. Capstone progress - Employee Attrition (initial EDA)", H1)]
    s += [Paragraph(f"{emp.shape[0]} employees; overall attrition rate {emp['attrition_flag'].mean() * 100:.1f}% (imbalanced target). "
                    "Missing income values were filled with the job-level median and missing satisfaction with the median. "
                    "<i>The dataset is synthetic - swap in the real HR file when available.</i>", BODY), Spacer(1, 6)]
    s += img("employee_attrition_by_category.png")
    s += img("employee_numeric_boxplots.png")
    s += img("employee_correlation_heatmap.png", 11.5 * cm)
    s += [Paragraph("Initial observations", H2)]
    s += bullets([
        f"Overtime is the clearest signal: {ot['Yes']:.1f}% attrition with overtime vs {ot['No']:.1f}% without.",
        f"Lower job satisfaction means higher attrition ({sat.loc[1]:.1f}% at level 1 vs {sat.loc[4]:.1f}% at level 4).",
        f"Income (r = {ec['MonthlyIncome']:.2f}) and tenure (r = {ec['YearsAtCompany']:.2f}) are negatively correlated with attrition.",
        f"Commute distance shows almost no linear relationship (r = {ec['DistanceFromHome']:.2f}).",
        "Age, tenure, job level and income are inter-correlated, so watch multicollinearity when modelling.",
    ])

    s += [Paragraph("5. Conclusion and next steps", H2),
          Paragraph("The environment, Python fundamentals, data cleaning and visualisation workflow are in place and documented. "
                    "Task 2 will use the cleaned datasets to preprocess features, split data, train baseline models "
                    "(logistic regression, random forest) and evaluate them using precision, recall and F1.", BODY)]

    out = ROOT / "reports" / "EDA_Report.pdf"
    out.parent.mkdir(exist_ok=True)
    doc = SimpleDocTemplate(str(out), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm,
                            topMargin=1.8 * cm, bottomMargin=1.8 * cm, title="AI/ML Task 1 - EDA Report")
    doc.build(s)
    print("Report written to", out)


if __name__ == "__main__":
    build()
