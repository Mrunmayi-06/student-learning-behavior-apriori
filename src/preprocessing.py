"""
OULAD preprocessing and transaction generation

Input : data/raw/*.csv
Output:
    data/cleaned_data.csv
    data/transactions.csv

The script uses early-stage behaviour (days 0-27 = Weeks 1-4),
then creates one transaction per student.
"""

from pathlib import Path
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
DATA_DIR = BASE_DIR / "data"

EARLY_START_DAY = 0
EARLY_END_DAY = 27          # first 4 weeks: days 0-27

# Activity types actually present in this OULAD copy.
FORUM_TYPES = {"forumng"}
QUIZ_TYPES = {"quiz", "externalquiz"}
RESOURCE_TYPES = {
    "resource", "subpage", "oucontent", "url", "page", "folder",
    "htmlactivity", "sharedsubpage", "repeatactivity"
}


def categorize_rank(series: pd.Series) -> pd.Series:
    """Split a numerical feature into Low/Medium/High using student ranks."""
    s = pd.to_numeric(series, errors="coerce").fillna(0)
    if s.nunique() <= 1:
        return pd.Series("Low", index=s.index)

    pct = s.rank(method="average", pct=True)
    return pd.cut(
        pct,
        bins=[-np.inf, 1 / 3, 2 / 3, np.inf],
        labels=["Low", "Medium", "High"],
        include_lowest=True,
    ).astype(str)


def load_student_base() -> pd.DataFrame:
    """Load the student-level tables and create the base student population."""
    info = pd.read_csv(
        RAW_DIR / "studentInfo.csv",
        usecols=["code_module", "code_presentation", "id_student", "final_result"],
    )

    registration = pd.read_csv(
        RAW_DIR / "studentRegistration.csv",
        usecols=[
            "code_module", "code_presentation", "id_student",
            "date_registration", "date_unregistration"
        ],
    )

    courses = pd.read_csv(
        RAW_DIR / "courses.csv",
        usecols=["code_module", "code_presentation", "module_presentation_length"],
    )

    base = info.merge(
        registration,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
        validate="one_to_one",
    )
    base = base.merge(
        courses,
        on=["code_module", "code_presentation"],
        how="left",
        validate="many_to_one",
    )

    # The project has three outcome categories. OULAD also has Distinction;
    # for this project it is grouped with the successful/pass outcome.
    base["final_outcome"] = base["final_result"].replace({"Distinction": "Pass"})
    base["final_outcome"] = base["final_outcome"].replace({
        "Pass": "Result_Pass",
        "Fail": "Result_Fail",
        "Withdrawn": "Result_Withdrawn",
    })

    # Keep only the three outcomes required by the project.
    base = base[base["final_outcome"].isin(
        ["Result_Pass", "Result_Fail", "Result_Withdrawn"]
    )].copy()

    return base


def extract_vle_features(base: pd.DataFrame) -> pd.DataFrame:
    """Aggregate early VLE behaviour for every student."""
    vle = pd.read_csv(RAW_DIR / "vle.csv")
    vle = vle[["id_site", "code_module", "code_presentation", "activity_type"]]

    # Read the large studentVle file in chunks to avoid loading ~10.6M rows
    # into memory at once.
    usecols = ["code_module", "code_presentation", "id_student", "id_site", "date", "sum_click"]
    student_keys = base[["code_module", "code_presentation", "id_student"]].copy()
    key_set = set(map(tuple, student_keys.itertuples(index=False, name=None)))

    parts = []
    for chunk in pd.read_csv(RAW_DIR / "studentVle.csv", usecols=usecols, chunksize=500_000):
        chunk = chunk[chunk["date"].between(EARLY_START_DAY, EARLY_END_DAY)]
        if chunk.empty:
            continue

        # Keep only students/courses in our student base.
        keys = list(zip(chunk["code_module"], chunk["code_presentation"], chunk["id_student"]))
        mask = [k in key_set for k in keys]
        chunk = chunk.loc[mask].copy()
        if chunk.empty:
            continue

        chunk = chunk.merge(
            vle,
            on=["id_site", "code_module", "code_presentation"],
            how="left",
            validate="many_to_one",
        )
        parts.append(chunk)

    if parts:
        activity = pd.concat(parts, ignore_index=True)
    else:
        activity = pd.DataFrame(columns=usecols + ["activity_type"])

    group_cols = ["code_module", "code_presentation", "id_student"]
    if activity.empty:
        features = base[group_cols].copy()
        features["total_clicks"] = 0
        features["active_days"] = 0
        features["forum_activity"] = 0
        features["quiz_activity"] = 0
        features["resource_activity"] = 0
        return features

    activity["sum_click"] = pd.to_numeric(activity["sum_click"], errors="coerce").fillna(0)
    activity["activity_type"] = activity["activity_type"].fillna("unknown")

    activity["forum_clicks"] = np.where(
        activity["activity_type"].isin(FORUM_TYPES), activity["sum_click"], 0
    )
    activity["quiz_clicks"] = np.where(
        activity["activity_type"].isin(QUIZ_TYPES), activity["sum_click"], 0
    )
    activity["resource_clicks"] = np.where(
        activity["activity_type"].isin(RESOURCE_TYPES), activity["sum_click"], 0
    )

    features = activity.groupby(group_cols).agg(
        total_clicks=("sum_click", "sum"),
        active_days=("date", "nunique"),
        forum_activity=("forum_clicks", "sum"),
        quiz_activity=("quiz_clicks", "sum"),
        resource_activity=("resource_clicks", "sum"),
    ).reset_index()

    return features


def extract_assignment_features(base: pd.DataFrame) -> pd.DataFrame:
    """Calculate early TMA/CMA submission timing for each student."""
    assessments = pd.read_csv(
        RAW_DIR / "assessments.csv",
        usecols=["code_module", "code_presentation", "id_assessment", "assessment_type", "date", "weight"],
    )
    student_assessment = pd.read_csv(
        RAW_DIR / "studentAssessment.csv",
        usecols=["id_assessment", "id_student", "date_submitted", "is_banked", "score"],
    )

    # Exams are not treated as assignment-timing behaviour.
    assessments = assessments[assessments["assessment_type"].isin(["TMA", "CMA"])].copy()
    assessments = assessments[assessments["date"].between(EARLY_START_DAY, EARLY_END_DAY)]

    # Which assessments were actually scheduled in the early period for each course.
    early_assessments = assessments[
        ["code_module", "code_presentation", "id_assessment", "date", "assessment_type"]
    ]

    if early_assessments.empty:
        out = base[["code_module", "code_presentation", "id_student"]].copy()
        out["assignment_timing"] = "Assignment_Missing"
        return out

    joined = base[["code_module", "code_presentation", "id_student"]].merge(
        early_assessments,
        on=["code_module", "code_presentation"],
        how="inner",
    )
    joined = joined.merge(student_assessment, on=["id_assessment", "id_student"], how="left")

    # A missing submission is treated as missing/not submitted.
    joined["delay"] = joined["date_submitted"] - joined["date"]

    rows = []
    for keys, g in joined.groupby(["code_module", "code_presentation", "id_student"]):
        submitted = g["date_submitted"].notna()
        if not submitted.any():
            category = "Assignment_Missing"
        else:
            # If any early assessment was not submitted, preserve that signal.
            if (~submitted).any():
                category = "Assignment_Missing"
            else:
                avg_delay = g.loc[submitted, "delay"].mean()
                if avg_delay < 0:
                    category = "Assignment_Early"
                elif avg_delay <= 1:
                    category = "Assignment_OnTime"
                else:
                    category = "Assignment_Late"
        rows.append((*keys, category))

    out = pd.DataFrame(
        rows,
        columns=["code_module", "code_presentation", "id_student", "assignment_timing"],
    )
    return out


def build_dataset() -> pd.DataFrame:
    base = load_student_base()
    vle_features = extract_vle_features(base)
    assignment_features = extract_assignment_features(base)

    df = base.merge(
        vle_features,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    ).merge(
        assignment_features,
        on=["code_module", "code_presentation", "id_student"],
        how="left",
    )

    numeric_features = [
        "total_clicks", "active_days", "forum_activity",
        "quiz_activity", "resource_activity"
    ]
    for col in numeric_features:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    df["assignment_timing"] = df["assignment_timing"].fillna("Assignment_Missing")

    # Convert numerical behaviour into categorical Apriori items.
    df["forum_category"] = "Forum_" + categorize_rank(df["forum_activity"])
    df["quiz_category"] = "Quiz_" + categorize_rank(df["quiz_activity"])
    df["resource_category"] = "Resource_" + categorize_rank(df["resource_activity"])
    df["active_days_category"] = "ActiveDays_" + categorize_rank(df["active_days"])

    # Overall engagement is based only on early behavioural measurements.
    # Rank-normalizing first prevents clicks from dominating active days.
    engagement_components = pd.DataFrame({
        "clicks": categorize_rank(df["total_clicks"]).map({"Low": 0, "Medium": 1, "High": 2}),
        "active_days": categorize_rank(df["active_days"]).map({"Low": 0, "Medium": 1, "High": 2}),
        "forum": categorize_rank(df["forum_activity"]).map({"Low": 0, "Medium": 1, "High": 2}),
        "quiz": categorize_rank(df["quiz_activity"]).map({"Low": 0, "Medium": 1, "High": 2}),
        "resource": categorize_rank(df["resource_activity"]).map({"Low": 0, "Medium": 1, "High": 2}),
    })
    engagement_score = engagement_components.mean(axis=1)
    df["engagement_category"] = pd.cut(
        engagement_score,
        bins=[-np.inf, 0.66, 1.33, np.inf],
        labels=["Engagement_Low", "Engagement_Medium", "Engagement_High"],
        include_lowest=True,
    ).astype(str)

    # Remove accidental duplicates at the student-course-presentation level.
    df = df.drop_duplicates(
        subset=["code_module", "code_presentation", "id_student"]
    ).copy()

    return df


def create_outputs(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    cleaned_columns = [
        "code_module", "code_presentation", "id_student",
        "date_registration", "date_unregistration", "module_presentation_length",
        "total_clicks", "active_days", "forum_activity", "quiz_activity",
        "resource_activity", "assignment_timing", "final_result", "final_outcome",
        "forum_category", "quiz_category", "resource_category",
        "active_days_category", "engagement_category",
    ]
    df[cleaned_columns].to_csv(DATA_DIR / "cleaned_data.csv", index=False)

    def make_transaction(row):
        return [
            row["forum_category"],
            row["quiz_category"],
            row["resource_category"],
            row["active_days_category"],
            row["assignment_timing"],
            row["engagement_category"],
            row["final_outcome"],
        ]

    transactions = df[["code_module", "code_presentation", "id_student"]].copy()
    transactions["items"] = df.apply(make_transaction, axis=1).map(lambda x: ",".join(x))
    transactions.to_csv(DATA_DIR / "transactions.csv", index=False)


def main():
    print("Loading OULAD and creating Member 1 features...")
    df = build_dataset()
    create_outputs(df)

    print(f"Students/transactions: {len(df):,}")
    print("Final outcomes:")
    print(df["final_outcome"].value_counts().to_string())
    print("\nOutputs created:")
    print(DATA_DIR / "cleaned_data.csv")
    print(DATA_DIR / "transactions.csv")


if __name__ == "__main__":
    main()
