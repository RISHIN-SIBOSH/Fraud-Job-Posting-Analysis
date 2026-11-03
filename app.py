from pathlib import Path

import altair as alt
import joblib
import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split

from predictor import MAX_POSTING_LENGTH, score_posting


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "fraud_job_detector.joblib"
DATA_PATH = BASE_DIR / "data" / "fake_job_postings.csv"

BACKGROUND = "#0D0E17"
CARD = "#191A28"
BORDER = "#2B2C3F"
TEXT = "#F4F4FB"
MUTED = "#A2A4B8"
PURPLE = "#7864F8"
PINK = "#F13B8B"
LIME = "#94DE51"
YELLOW = "#F5C34D"
TEAL = "#52D1C5"

st.set_page_config(
    page_title="Job Posting Fraud Analyzer",
    page_icon=":material/verified_user:",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_bundle(path):
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_validation_support(path):
    frame = pd.read_csv(path, low_memory=False)
    frame = frame.replace(r"^\s*$", pd.NA, regex=True)
    frame = frame.drop_duplicates(
        subset=["title", "description", "company_profile"]
    )
    frame = frame.drop_duplicates(subset=["description"]).copy()
    frame["fraudulent"] = pd.to_numeric(
        frame["fraudulent"], errors="raise"
    ).astype(int)
    _, temporary = train_test_split(
        frame.drop(columns=["job_id"]),
        test_size=0.30,
        stratify=frame["fraudulent"],
        random_state=42,
    )
    validation, _ = train_test_split(
        temporary,
        test_size=0.50,
        stratify=temporary["fraudulent"],
        random_state=42,
    )
    counts = validation["fraudulent"].value_counts()
    return int(counts.get(0, 0)), int(counts.get(1, 0))


@st.cache_data(show_spinner="Loading the dataset...")
def load_dataset(path):
    frame = pd.read_csv(path, low_memory=False)
    if "fraudulent" not in frame.columns:
        raise ValueError("The dataset does not contain the 'fraudulent' label column.")

    frame["fraudulent"] = pd.to_numeric(frame["fraudulent"], errors="coerce")
    frame = frame.dropna(subset=["fraudulent"]).copy()
    frame["fraudulent"] = frame["fraudulent"].astype(int)
    return frame


def render_page_header(eyebrow, title, description, icon):
    st.markdown(
        f'<div class="eyebrow">{eyebrow}</div>',
        unsafe_allow_html=True,
    )
    st.title(title, icon=icon)
    st.caption(description)


def style_chart(chart):
    return (
        chart.configure(font="Poppins")
        .configure_axis(
            labelFont="Poppins",
            titleFont="Poppins",
            labelFontSize=11,
            titleFontSize=11,
            labelColor=MUTED,
            titleColor="#D9DAE7",
            domainColor="#3A3B50",
            tickColor="#3A3B50",
            gridColor="#292A3B",
            gridOpacity=0.7,
        )
        .configure_view(stroke="transparent")
        .configure(background="transparent")
    )


def render_gradient_chart(data, category, value, description, percentage=False):
    if data.empty:
        st.info("There is not enough data to display this chart.")
        return

    maximum = float(data[value].max()) if len(data) else 0.0
    axis_format = ".0%" if percentage else ",.0f"
    chart = (
        alt.Chart(data)
        .mark_bar(cornerRadiusTopLeft=7, cornerRadiusTopRight=7)
        .encode(
            x=alt.X(
                f"{category}:N",
                sort="-y",
                axis=alt.Axis(labelAngle=-20, labelLimit=160, title=None),
            ),
            y=alt.Y(
                f"{value}:Q",
                axis=alt.Axis(format=axis_format, title=None),
                scale=alt.Scale(domainMin=0),
            ),
            color=alt.Color(
                f"{value}:Q",
                scale=alt.Scale(
                    domain=[0, maximum or 1],
                    range=["#5747A7", PINK],
                ),
                legend=None,
            ),
            tooltip=[
                alt.Tooltip(f"{category}:N", title=category),
                alt.Tooltip(f"{value}:Q", title=value, format=axis_format),
            ],
        )
        .properties(height=270)
    )
    st.altair_chart(style_chart(chart), alt=description)


def category_values(series):
    values = series.astype("string").str.strip()
    values = values.fillna("Not specified").replace("", "Not specified")
    return values.astype(str)


def rate_summary(frame, categories, label, min_postings=20, limit=10):
    working = pd.DataFrame(
        {
            "Category": category_values(categories),
            "Fraud": frame["fraudulent"].eq(1).astype(int),
        }
    )
    summary = (
        working.groupby("Category", as_index=False)
        .agg(
            Postings=("Fraud", "size"),
            **{"Fraud rate": ("Fraud", "mean")},
        )
    )
    summary = summary[summary["Postings"] >= min_postings]
    summary = summary.sort_values("Fraud rate", ascending=False).head(limit)
    return summary.rename(columns={"Category": label})


def show_chart_card(
    title,
    icon,
    data,
    category,
    value,
    description,
    percentage=False,
    note=None,
):
    with st.container(border=True):
        st.subheader(title, icon=icon)
        render_gradient_chart(
            data,
            category,
            value,
            description,
            percentage=percentage,
        )
        if note:
            st.caption(note)


st.markdown(
    """
    <style>
    .stApp {
        background:
            radial-gradient(ellipse at 82% -8%, rgba(120, 100, 248, 0.20), transparent 42%),
            radial-gradient(ellipse at 0% 28%, rgba(241, 59, 139, 0.07), transparent 28%),
            #0D0E17;
        color: #F4F4FB;
    }

    [data-testid="stHeader"] {
        background: rgba(13, 14, 23, 0.88);
    }

    [data-testid="stSidebar"] {
        background: #151622;
        border-right: 1px solid #292A3B;
    }

    [data-testid="stSidebar"] > div:first-child {
        padding-top: 0.8rem;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        padding: 0.8rem 0.55rem 1rem;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a {
        border: 1px solid transparent;
        border-radius: 12px;
        margin: 0.28rem 0;
        padding: 0.62rem 0.75rem;
        color: #C4C5D3;
        transition:
            background-color 260ms cubic-bezier(0.22, 1, 0.36, 1),
            border-color 260ms cubic-bezier(0.22, 1, 0.36, 1),
            transform 320ms cubic-bezier(0.22, 1, 0.36, 1);
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a:hover {
        background: rgba(120, 100, 248, 0.13);
        border-color: rgba(120, 100, 248, 0.30);
        transform: translateX(3px);
        color: #FFFFFF;
    }

    [data-testid="stSidebar"] [data-testid="stSidebarNav"] a[aria-current="page"] {
        background: linear-gradient(110deg, rgba(120, 100, 248, 0.22), rgba(241, 59, 139, 0.10));
        border-color: rgba(143, 128, 255, 0.42);
        box-shadow: inset 3px 0 #F13B8B, 0 8px 22px rgba(0, 0, 0, 0.16);
        color: #FFFFFF;
        font-weight: 600;
    }

    .block-container {
        max-width: 1480px;
        padding-top: 3.1rem;
        padding-bottom: 3.5rem;
        animation: reveal 520ms cubic-bezier(0.22, 1, 0.36, 1) both;
    }

    @keyframes reveal {
        from { opacity: 0; transform: translateY(8px); }
        to { opacity: 1; transform: translateY(0); }
    }

    h1 {
        color: #F8F7FC;
        font-weight: 700;
        letter-spacing: -0.035em;
        line-height: 1.25;
        padding-top: 0.08em;
    }

    h2, h3, h4 {
        color: #F0EFF8;
        font-weight: 600;
        letter-spacing: -0.015em;
    }

    .eyebrow {
        display: block;
        width: fit-content;
        color: #B2A6FF;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.16em;
        line-height: 1.75;
        margin: 0.25rem 0 0.55rem;
        padding: 0.12rem 0.12rem 0.18rem;
        overflow: visible;
        text-transform: uppercase;
    }

    .sidebar-brand {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        padding: 0.6rem 0.9rem 1.2rem;
        margin: 0 0.55rem 0.8rem;
        border-bottom: 1px solid #2A2B3D;
    }

    .sidebar-brand-mark {
        width: 42px;
        height: 42px;
        display: grid;
        place-items: center;
        border-radius: 13px;
        background: linear-gradient(140deg, #7864F8, #F13B8B);
        box-shadow: 0 8px 24px rgba(120, 100, 248, 0.24);
        color: #FFFFFF;
        font-size: 1rem;
        font-weight: 700;
    }

    .sidebar-brand-title {
        color: #F6F5FB;
        font-size: 0.92rem;
        font-weight: 600;
        line-height: 1.25;
    }

    .sidebar-brand-caption {
        color: #9698AD;
        font-size: 0.64rem;
        font-weight: 600;
        letter-spacing: 0.13em;
        margin-top: 0.28rem;
    }

    .selected-model-name {
        color: #F8F7FC;
        font-size: clamp(1.1rem, 1.7vw, 1.55rem);
        font-weight: 700;
        letter-spacing: -0.025em;
        line-height: 1.3;
        overflow-wrap: anywhere;
        padding-top: 0.35rem;
    }

    [data-testid="stMetric"] {
        background: rgba(25, 26, 40, 0.92);
        border: 1px solid #2B2C40;
        border-radius: 16px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.15);
        transition:
            transform 450ms cubic-bezier(0.22, 1, 0.36, 1),
            border-color 350ms ease,
            box-shadow 450ms cubic-bezier(0.22, 1, 0.36, 1),
            filter 400ms ease,
            opacity 400ms ease;
    }

    [data-testid="stVerticalBlockBorderWrapper"] {
        position: relative;
        z-index: 1;
        overflow: visible;
        border: 1px solid rgba(95, 94, 132, 0.34);
        border-radius: 18px;
        background: rgba(25, 26, 40, 0.90);
        box-shadow: 0 9px 26px rgba(0, 0, 0, 0.14);
        transition:
            transform 460ms cubic-bezier(0.22, 1, 0.36, 1),
            border-color 360ms ease,
            box-shadow 460ms cubic-bezier(0.22, 1, 0.36, 1),
            filter 400ms ease,
            opacity 400ms ease;
    }

    [data-testid="stTextArea"] textarea {
        background: #11121D;
        border: 1px solid #36374D;
        border-radius: 13px;
        color: #F4F4FB;
        line-height: 1.65;
    }

    [data-testid="stTextArea"] textarea:focus {
        border-color: #8A78FF;
        box-shadow: 0 0 0 1px rgba(138, 120, 255, 0.7);
    }

    .stButton button[kind="primary"] {
        border-radius: 11px;
        font-weight: 600;
        box-shadow: 0 7px 22px rgba(120, 100, 248, 0.22);
        transition:
            transform 260ms cubic-bezier(0.22, 1, 0.36, 1),
            box-shadow 260ms ease;
    }

    .stButton button[kind="primary"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 11px 27px rgba(120, 100, 248, 0.34);
    }

    [data-testid="stDataFrame"] {
        border: 1px solid #2C2D41;
        border-radius: 12px;
        overflow: hidden;
    }

    @media (hover: hover) {
        [data-testid="stMainBlockContainer"]:has(
            :is([data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"]):hover
        ) :is([data-testid="stVerticalBlockBorderWrapper"], [data-testid="stMetric"]):not(:hover) {
            filter: blur(1.3px);
            opacity: 0.62;
        }

        [data-testid="stVerticalBlockBorderWrapper"]:hover {
            z-index: 20;
            transform: translate3d(0, -4px, 0) scale(1.018);
            border-color: rgba(143, 128, 255, 0.62);
            box-shadow: 0 19px 42px rgba(0, 0, 0, 0.32), 0 0 22px rgba(120, 100, 248, 0.12);
        }

        [data-testid="stMetric"]:hover {
            z-index: 20;
            transform: translate3d(0, -3px, 0) scale(1.025);
            border-color: rgba(143, 128, 255, 0.68);
            box-shadow: 0 17px 35px rgba(0, 0, 0, 0.30);
        }
    }

    @media (max-width: 760px) {
        .block-container {
            padding: 2.25rem 1rem 2.5rem;
        }

        h1 {
            font-size: 1.85rem;
        }
    }

    @media (prefers-reduced-motion: reduce) {
        *, *::before, *::after {
            animation-duration: 0.01ms !important;
            transition-duration: 0.01ms !important;
        }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


try:
    bundle = load_bundle(MODEL_PATH)
    model = bundle["model"]
    threshold = float(bundle.get("threshold", 0.50))
    selected_model_name = bundle.get("selected_model", "Calibrated Linear SVM")
    model_comparison = bundle.get("model_comparison", [])
    selection_reason = bundle.get("selection_reason", "")
except FileNotFoundError:
    st.error(f"Model file not found: {MODEL_PATH}")
    st.stop()
except Exception as exc:
    st.error(f"Could not load the model bundle: {exc}")
    st.stop()


def analyze_posting_page():
    render_page_header(
        "Fraud screening",
        "Fake job testing",
        "Paste a complete job listing to get a screening score for human review.",
        ":material/search:",
    )

    input_col, detail_col = st.columns([1.8, 0.9], gap="large")
    with input_col:
        with st.container(border=True):
            st.subheader("Paste a job listing", icon=":material/article:")
            with st.form("job_posting_form", border=False):
                full_posting = st.text_area(
                    "Complete job posting",
                    height=300,
                    max_chars=MAX_POSTING_LENGTH,
                    placeholder="Paste the complete job description here...",
                    help=(
                        "Include the title, company details, description, "
                        "requirements, and benefits when available."
                    ),
                )
                submitted = st.form_submit_button(
                    "Analyze posting",
                    type="primary",
                    icon=":material/search:",
                )

    with detail_col:
        with st.container(border=True):
            st.subheader("Screening setup", icon=":material/tune:")
            st.metric("Human-review threshold", f"{threshold:.0%}")
            st.caption(f"Active classifier: {selected_model_name}")
            st.markdown(
                "A score at or above the threshold is **flagged for review**. "
                "The result is a model estimate, not proof of fraud."
            )
        with st.container(border=True):
            st.subheader("How to use", icon=":material/info:")
            st.markdown(
                "1. Copy the complete job listing.\n"
                "2. Paste it into the box.\n"
                "3. Review the score and verify the employer independently."
            )

    if submitted:
        st.session_state.pop("last_screening_result", None)
        try:
            fraud_score, flagged = score_posting(model, full_posting, threshold)
            st.session_state["last_screening_result"] = {
                "score": fraud_score,
                "flagged": flagged,
            }
        except (TypeError, ValueError) as exc:
            st.warning(str(exc))
        except Exception as exc:
            st.error(f"Could not analyze this posting: {exc}")

    result = st.session_state.get("last_screening_result")
    if result:
        with st.container(border=True):
            st.subheader("Screening result", icon=":material/assessment:")
            score_col, result_col = st.columns([0.75, 1.5], gap="large")
            with score_col:
                st.metric("Estimated fraud score", f"{result['score']:.1%}")
                st.progress(result["score"])
            with result_col:
                if result["flagged"]:
                    st.error(
                        f"Flagged for review: score is at or above "
                        f"{threshold:.0%}."
                    )
                    st.write("Review the listing and verify the employer.")
                else:
                    st.success("Not flagged by the model.")
                    st.write(
                        "A score below the threshold does not prove the "
                        "listing is genuine."
                    )
            st.caption(
                "Automated screening only. Verify employers independently. "
                "Pasted posting text is not saved."
            )


def eda_dashboard_page():
    render_page_header(
        "Exploratory data analysis",
        "Dataset dashboard",
        "Historical patterns in the labeled job-posting dataset used by the project.",
        ":material/analytics:",
    )

    if not DATA_PATH.exists():
        st.error(f"Dataset file not found: {DATA_PATH}")
        st.caption("Job screening and model comparison remain available.")
        return

    try:
        df = load_dataset(str(DATA_PATH))
    except Exception as exc:
        st.error(f"Could not load the dataset: {exc}")
        return

    total = len(df)
    fraudulent = int(df["fraudulent"].eq(1).sum())
    genuine = int(df["fraudulent"].eq(0).sum())
    fraud_rate = fraudulent / total if total else 0.0

    metric_cols = st.columns(4, gap="medium")
    metric_cols[0].metric("Total postings", f"{total:,}")
    metric_cols[1].metric("Genuine", f"{genuine:,}")
    metric_cols[2].metric("Fraudulent", f"{fraudulent:,}")
    metric_cols[3].metric("Fraud rate", f"{fraud_rate:.1%}")

    label_data = (
        df["fraudulent"]
        .map({0: "Genuine", 1: "Fraudulent"})
        .value_counts()
        .rename_axis("Posting label")
        .reset_index(name="Postings")
    )

    employment_values = category_values(
        df["employment_type"]
        if "employment_type" in df.columns
        else pd.Series(dtype="string")
    )
    employment_data = (
        employment_values.value_counts()
        .rename_axis("Employment type")
        .reset_index(name="Postings")
    )

    employment_rate = (
        rate_summary(df, df["employment_type"], "Employment type")
        if "employment_type" in df.columns
        else pd.DataFrame()
    )

    remote_data = pd.DataFrame()
    if "telecommuting" in df.columns:
        remote = pd.to_numeric(df["telecommuting"], errors="coerce").fillna(0)
        remote_data = (
            remote.map(
                lambda value: "Remote" if value == 1 else "Not marked remote"
            )
            .value_counts()
            .rename_axis("Work arrangement")
            .reset_index(name="Postings")
        )

    experience_rate = (
        rate_summary(df, df["required_experience"], "Experience")
        if "required_experience" in df.columns
        else pd.DataFrame()
    )
    education_rate = (
        rate_summary(df, df["required_education"], "Education")
        if "required_education" in df.columns
        else pd.DataFrame()
    )
    industry_rate = (
        rate_summary(df, df["industry"], "Industry", limit=10)
        if "industry" in df.columns
        else pd.DataFrame()
    )

    logo_rate = pd.DataFrame()
    if "has_company_logo" in df.columns:
        logo_flag = pd.to_numeric(
            df["has_company_logo"], errors="coerce"
        ).fillna(0)
        logo_labels = logo_flag.map(
            lambda value: "Logo shown" if value == 1 else "No logo shown"
        )
        logo_rate = rate_summary(df, logo_labels, "Company profile")

    chart_specs = [
        (
            "Posting labels",
            ":material/bar_chart:",
            label_data,
            "Posting label",
            "Postings",
            "Distribution of genuine and fraudulent labels in the dataset.",
            False,
            None,
        ),
        (
            "Employment type",
            ":material/work:",
            employment_data,
            "Employment type",
            "Postings",
            "Historical posting counts by employment type.",
            False,
            None,
        ),
        (
            "Fraud rate by employment type",
            ":material/query_stats:",
            employment_rate,
            "Employment type",
            "Fraud rate",
            "Fraud rate grouped by employment type; categories need at least 20 postings.",
            True,
            "Categories with fewer than 20 postings are omitted.",
        ),
        (
            "Remote-work postings",
            ":material/laptop:",
            remote_data,
            "Work arrangement",
            "Postings",
            "Historical counts of postings marked remote or not marked remote.",
            False,
            None,
        ),
        (
            "Fraud rate by experience",
            ":material/work_history:",
            experience_rate,
            "Experience",
            "Fraud rate",
            "Fraud rate grouped by required experience.",
            True,
            "Categories with fewer than 20 postings are omitted.",
        ),
        (
            "Fraud rate by education",
            ":material/school:",
            education_rate,
            "Education",
            "Fraud rate",
            "Fraud rate grouped by required education.",
            True,
            "Categories with fewer than 20 postings are omitted.",
        ),
        (
            "Fraud rate by industry",
            ":material/domain:",
            industry_rate,
            "Industry",
            "Fraud rate",
            "Ten industries with the highest fraud rate among groups with at least 20 postings.",
            True,
            "Only the ten highest-rate groups are shown.",
        ),
        (
            "Fraud rate by company logo",
            ":material/verified:",
            logo_rate,
            "Company profile",
            "Fraud rate",
            "Fraud rate grouped by whether the posting includes a company logo.",
            True,
            None,
        ),
    ]

    for start in range(0, len(chart_specs), 2):
        left, right = st.columns(2, gap="large")
        for column, spec in zip((left, right), chart_specs[start : start + 2]):
            title, icon, data, category, value, alt, percentage, note = spec
            with column:
                show_chart_card(
                    title,
                    icon,
                    data,
                    category,
                    value,
                    alt,
                    percentage=percentage,
                    note=note,
                )

    st.caption(
        "LinkedIn or current job-market statistics."
    )


def comparison_charts(comparison):
    pr_auc_chart = (
        alt.Chart(comparison)
        .mark_bar(cornerRadiusTopRight=7, cornerRadiusBottomRight=7)
        .encode(
            x=alt.X(
                "validation_pr_auc:Q",
                title="Validation PR-AUC",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".2f"),
            ),
            y=alt.Y("model:N", title=None, sort="-x"),
            color=alt.condition(
                alt.datum.selected,
                alt.value(PINK),
                alt.value(PURPLE),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip(
                    "validation_pr_auc:Q",
                    title="Validation PR-AUC",
                    format=".3f",
                ),
            ],
        )
        .properties(height=260)
    )

    points = (
        alt.Chart(comparison)
        .mark_circle(opacity=0.92, stroke="#F4F4FB", strokeWidth=0.8)
        .encode(
            x=alt.X(
                "fraud_precision:Q",
                title="Fraud precision",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            y=alt.Y(
                "fraud_recall:Q",
                title="Fraud recall",
                scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format=".0%"),
            ),
            size=alt.Size(
                "fraud_f1:Q",
                title="Fraud F1",
                scale=alt.Scale(range=[160, 520]),
                legend=None,
            ),
            color=alt.condition(
                alt.datum.selected,
                alt.value(PINK),
                alt.value(PURPLE),
            ),
            tooltip=[
                alt.Tooltip("model:N", title="Model"),
                alt.Tooltip("fraud_precision:Q", title="Precision", format=".3f"),
                alt.Tooltip("fraud_recall:Q", title="Recall", format=".3f"),
                alt.Tooltip("fraud_f1:Q", title="F1", format=".3f"),
                alt.Tooltip(
                    "genuine_jobs_flagged:Q",
                    title="Genuine jobs flagged",
                ),
                alt.Tooltip("frauds_missed:Q", title="Frauds missed"),
            ],
        )
        .properties(height=260)
    )
    labels = (
        alt.Chart(comparison)
        .mark_text(
            dy=-17,
            font="Poppins",
            fontSize=10,
            fontWeight=600,
            color=TEXT,
        )
        .encode(
            x="fraud_precision:Q",
            y="fraud_recall:Q",
            text="model:N",
        )
    )

    return style_chart(pr_auc_chart), style_chart(points + labels)


def model_comparison_page():
    render_page_header(
        "Model evaluation",
        "Model comparison",
        "Three classifiers trained on the same split and compared on validation data.",
        ":material/leaderboard:",
    )

    comparison = pd.DataFrame(model_comparison)
    if comparison.empty:
        st.warning("Model-comparison results are not available in the saved model bundle.")
        return

    selected_rows = comparison[comparison["selected"]]
    selected_metrics = (
        selected_rows.iloc[0]
        if not selected_rows.empty
        else comparison.sort_values("validation_pr_auc", ascending=False).iloc[0]
    )

    genuine_support, fraud_support = load_validation_support(DATA_PATH)
    validation_total = genuine_support + fraud_support
    comparison["validation_accuracy"] = comparison.apply(
        lambda row: (
            genuine_support - int(row["genuine_jobs_flagged"])
            + fraud_support - int(row["frauds_missed"])
        ) / validation_total,
        axis=1,
    )

    summary_cols = st.columns([1.5, 1, 1, 1], gap="medium")
    with summary_cols[0]:
        with st.container(border=True):
            st.caption("Selected model")
            st.markdown(
                f'<div class="selected-model-name">{selected_model_name}</div>',
                unsafe_allow_html=True,
            )
    summary_cols[1].metric(
        "Validation PR-AUC",
        f"{selected_metrics['validation_pr_auc']:.3f}",
    )
    summary_cols[2].metric("Fraud F1", f"{selected_metrics['fraud_f1']:.3f}")
    summary_cols[3].metric("Review threshold", f"{threshold:.0%}")

    pr_chart, tradeoff_chart = comparison_charts(comparison)
    left, right = st.columns(2, gap="large")
    with left:
        with st.container(border=True):
            st.subheader("Validation PR-AUC", icon=":material/bar_chart:")
            st.altair_chart(
                pr_chart,
                alt="Validation precision-recall area under the curve for each trained model.",
            )
            st.caption("Higher values indicate stronger ranking on the validation set.")
    with right:
        with st.container(border=True):
            st.subheader("Fraud precision and recall", icon=":material/scatter_plot:")
            st.altair_chart(
                tradeoff_chart,
                alt="Model validation precision versus recall; larger circles have higher F1.",
            )
            st.caption("Larger circles indicate a higher fraud F1-score.")

    with st.container(border=True):
        st.subheader("Validation results", icon=":material/table_chart:")
        st.caption(
            "All three models use the same 50% threshold and validation split. "
            "Accuracy is shown with fraud-focused metrics because genuine jobs "
            "make up most of this dataset."
        )
        table = comparison.rename(
            columns={
                "model": "Classifier",
                "selected": "Decision",
                "validation_pr_auc": "Validation PR-AUC",
                "threshold": "Threshold",
                "fraud_precision": "Fraud precision",
                "fraud_recall": "Fraud recall",
                "fraud_f1": "Fraud F1",
                "genuine_jobs_flagged": "Genuine jobs flagged",
                "frauds_missed": "Frauds missed",
            }
        ).copy()
        table["Decision"] = table["Decision"].map(
            {True: "Selected", False: "Compared"}
        )
        st.dataframe(
            table[
                [
                    "Classifier",
                    "Decision",
                    "Validation PR-AUC",
                    "Validation accuracy",
                    "Threshold",
                    "Fraud precision",
                    "Fraud recall",
                    "Fraud F1",
                    "Genuine jobs flagged",
                    "Frauds missed",
                ]
            ],
            hide_index=True,
            width="stretch",
            alt="Validation results for the three trained fraud classifiers.",
            column_config={
                "Validation PR-AUC": st.column_config.NumberColumn(
                    "Validation PR-AUC",
                    format="%.3f",
                ),
                "Validation accuracy": st.column_config.NumberColumn(
                    "Validation accuracy",
                    format="percent",
                ),
                "Threshold": st.column_config.NumberColumn(
                    "Threshold",
                    format="percent",
                ),
                "Fraud precision": st.column_config.NumberColumn(
                    "Fraud precision",
                    format="%.3f",
                ),
                "Fraud recall": st.column_config.NumberColumn(
                    "Fraud recall",
                    format="%.3f",
                ),
                "Fraud F1": st.column_config.NumberColumn(
                    "Fraud F1",
                    format="%.3f",
                ),
            },
        )

    with st.container(border=True):
        st.subheader("Why this model was selected", icon=":material/check_circle:")
        st.success(f"{selected_model_name} is the active screening model.")
        st.write(selection_reason)
        st.caption(
            "This choice was made using validation results. The held-out test "
            "split was not used for model selection, and the score is not a "
            "calibrated real-world probability."
        )


with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-brand">
            <div class="sidebar-brand-mark">JF</div>
            <div>
                <div class="sidebar-brand-title">Job Posting Analyzer</div>
                <div class="sidebar-brand-caption">FRAUD SCREENING SUITE</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


page = st.navigation(
    [
        st.Page(
            analyze_posting_page,
            title="Fake job testing",
            icon=":material/search:",
            default=True,
        ),
        st.Page(
            eda_dashboard_page,
            title="EDA dashboard",
            icon=":material/analytics:",
        ),
        st.Page(
            model_comparison_page,
            title="Model comparison",
            icon=":material/leaderboard:",
        ),
    ],
    position="sidebar",
)
page.run()
