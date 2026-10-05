# Student Learning Behavior & Academic Outcome Pattern Mining

This project analyzes student learning behavior and academic outcomes using **Association Rule Mining with the Apriori algorithm**.

The project uses the **Open University Learning Analytics Dataset (OULAD)** to identify relationships between students' early-stage learning activities and their final academic outcomes.

## Project Objective

The objective is to discover meaningful associations between:

- Forum activity
- Quiz activity
- Resource/video activity
- Active learning days
- Assignment submission timing
- Overall engagement
- Final academic outcome

The project uses the **Apriori algorithm** to generate frequent itemsets and association rules from student learning transactions.

> **Note:** This project focuses on association patterns and does not make causal or predictive claims.

---

## Dataset

This project uses the **Open University Learning Analytics Dataset (OULAD)**.

The dataset is available on Kaggle:

**Dataset:** Student Demographics & Online Education Data (OULAD)  
https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad

The raw OULAD dataset is **not included in this repository** because some files are very large and exceed GitHub's file-size limits.

### Downloading the Dataset

1. Download the OULAD dataset from the Kaggle link above.
2. Extract the downloaded files.
3. Place the CSV files inside:

```text
data/raw/
