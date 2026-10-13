# Student Learning Behavior & Academic Outcome Pattern Mining

This project analyzes student learning patterns using association rule mining with the Apriori algorithm. It uses the Open University Learning Analytics Dataset (OULAD) to discover relationships between early learning behavior and final academic outcomes such as pass, fail, and withdrawn.

The project is designed to answer questions like:

- Which engagement patterns are associated with successful outcomes?
- Do students who submit assignments early tend to pass more often?
- How do forum, quiz, resource, and active-day behavior relate to final results?

> This project focuses on association patterns and not causal or predictive claims.

## Project goals

The workflow combines data preprocessing, feature engineering, transaction generation, Apriori mining, and result visualization.

Key behaviors included in the mining process:

- Forum activity
- Quiz activity
- Resource and VLE activity
- Active learning days
- Assignment submission timing
- Overall engagement level
- Final academic outcome

## Repository structure

```text
.
├── app.py                     # Streamlit dashboard
├── main.py                    # CLI analysis entry point
├── requirements.txt           # Python dependencies
├── data/
│   ├── cleaned_data.csv       # processed student feature table
│   ├── transactions.csv       # base transaction file
│   └── raw/                   # OULAD CSV files (download separately)
├── results/
│   └── analysis/              # generated itemsets, rules, and plots
├── src/
│   ├── analysis.py            # mining, comparisons, plotting
│   ├── apriori.py            # Apriori implementation
│   ├── preprocessing.py      # OULAD cleaning and transaction generation
│   ├── rules.py              # association rule generation
│   └── metrics.py            # additional metric utilities
└── README.md
```

## Dataset

This project uses the OULAD dataset from Kaggle:

https://www.kaggle.com/datasets/anlgrbz/student-demographics-online-education-dataoulad

The raw files are not stored in this repository because they are large. You must download them separately and place the extracted CSV files in:

```text
data/raw/
```

The required OULAD CSV files include items such as:

- studentInfo.csv
- studentRegistration.csv
- courses.csv
- assessments.csv
- studentAssessment.csv
- vle.csv
- studentVle.csv

## Setup

Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Running the project

### 1) Build the dataset and transaction files

From the project root:

```bash
python src/preprocessing.py
```

This creates:

- `data/cleaned_data.csv`
- `data/transactions.csv`

### 2) Run the analysis pipeline

```bash
python main.py
```

This loads the transaction file, mines frequent itemsets and association rules, evaluates threshold settings, and saves outputs into `results/analysis/`.

To also generate period-based transaction files for Weeks 1-8 and Full-course comparison:

```bash
python main.py --periods
```

### 3) Launch the Streamlit dashboard

```bash
streamlit run app.py
```

The dashboard allows you to:

- choose a period or dataset slice
- adjust support and confidence thresholds
- filter by outcome
- inspect frequent itemsets and rules
- review saved graphs

## Output files

The project writes generated outputs to the `results/` directory, including:

- `results/analysis/frequent_itemsets.csv`
- `results/analysis/all_rules.csv`
- `results/analysis/rules_pass.csv`
- `results/analysis/rules_fail.csv`
- `results/analysis/rules_withdrawn.csv`
- threshold experiment and comparison CSVs
- PNG charts for rule and support analysis

## Data preprocessing approach

The preprocessing stage converts raw OULAD records into student transactions:

- filters students to the three outcomes used in this study: Pass, Fail, Withdrawn
- aggregates VLE activity into forum, quiz, resource, and active-day features
- assesses assignment timing behavior (early, on time, late, or missing)
- ranks engagement signals into low, medium, and high categories
- creates one transaction per student containing behavior items and the final outcome

## Notes

- The project is intended for educational data mining and pattern discovery.
- It does not claim to establish causality between behavior and outcomes.
- Large OULAD source files must be downloaded externally and are not bundled with the repository.

## License

This project does not include a formal license file. Use it for educational and research purposes in line with the OULAD dataset terms and your local usage requirements.
