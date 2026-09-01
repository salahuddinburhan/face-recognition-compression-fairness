from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Face Recognition Fairness",
    page_icon="⚖️",
    layout="wide",
)

st.markdown(
    """
    <style>

    /* ======================================================
       PAGE LAYOUT
       ====================================================== */

    .block-container {
        max-width: 1380px;
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        padding-left: 2.5rem;
        padding-right: 2.5rem;
    }

    /* Reduce excessive vertical gaps between Streamlit blocks */
    [data-testid="stVerticalBlock"] {
        gap: 0.8rem;
    }


    /* ======================================================
       TYPOGRAPHY
       ====================================================== */

    h1 {
        font-size: 2.55rem !important;
        line-height: 1.15 !important;
        letter-spacing: -0.03em;
        margin-bottom: 0.35rem !important;
    }

    /* Major Streamlit section headings */
    h2 {
        font-size: 1.8rem !important;
        line-height: 1.25 !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
        margin-top: 0.5rem !important;
        margin-bottom: 0.5rem !important;
    }

    h3 {
        font-size: 1.75rem !important;
        line-height: 1.25 !important;
        font-weight: 700 !important;
        letter-spacing: -0.015em;
        margin-top: 0.45rem !important;
        margin-bottom: 0.45rem !important;
    }

    h4, h5 {
        margin-top: 0.3rem !important;
        margin-bottom: 0.25rem !important;
    }

    p {
        line-height: 1.65;
    }


    /* ======================================================
       SIDEBAR
       ====================================================== */

    section[data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
        padding-left: 1.3rem;
        padding-right: 1.3rem;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        margin-bottom: 0.8rem !important;
    }

    /* Select boxes */
    section[data-testid="stSidebar"] [data-baseweb="select"] > div {
        border-radius: 10px;
    }


    /* ======================================================
       KPI / METRIC CARDS
       ====================================================== */

    [data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.22);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        background: rgba(128, 128, 128, 0.035);
        min-height: 112px;
    }

    [data-testid="stMetricLabel"] {
        font-size: 0.88rem !important;
        font-weight: 600 !important;
        opacity: 0.82;
    }

    [data-testid="stMetricValue"] {
        font-size: 1.95rem !important;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
    }


    /* ======================================================
       INFO / KEY TAKEAWAY BOXES
       ====================================================== */

    [data-testid="stAlert"] {
        border-radius: 12px;
        padding-top: 0.0rem;
        padding-bottom: 0.0rem;
        border: 1px solid rgba(128, 128, 128, 0.18);
    }

    [data-testid="stAlert"] p {
        line-height: 1.55;
    }


    /* ======================================================
       PLOTLY CHART CONTAINERS
       ====================================================== */

    [data-testid="stPlotlyChart"] {
        border: 1px solid rgba(128, 128, 128, 0.16);
        border-radius: 14px;
        padding: 0.5rem;
        overflow: hidden;
    }


    /* ======================================================
       EXPANDERS
       ====================================================== */

    [data-testid="stExpander"] {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 12px;
        overflow: hidden;
    }

    [data-testid="stExpander"] summary {
        font-weight: 600;
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
    }


    /* ======================================================
       DATAFRAME
       ====================================================== */

    [data-testid="stDataFrame"] {
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid rgba(128, 128, 128, 0.15);
    }


    /* ======================================================
       CAPTIONS
       ====================================================== */

    [data-testid="stCaptionContainer"] {
        opacity: 0.78;
        font-size: 0.88rem;
    }


    /* ======================================================
       DIVIDERS
       ====================================================== */

    hr {
        margin-top: 2.2rem !important;
        margin-bottom: 2rem !important;
        opacity: 0.28;
    }


    /* ======================================================
       LINKS
       ====================================================== */

    a {
        text-decoration: none;
    }

    a:hover {
        text-decoration: underline;
    }


    /* ======================================================
       MOBILE / SMALL SCREENS
       ====================================================== */

    @media (max-width: 768px) {

        .block-container {
            padding-left: 1rem;
            padding-right: 1rem;
            padding-top: 1.2rem;
        }

        h1 {
            font-size: 2rem !important;
        }

        h2 {
            font-size: 1.45rem !important;
        }

        [data-testid="stMetricValue"] {
            font-size: 1.65rem !important;
        }
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# PROJECT PATHS
# ============================================================

APP_DIR = Path(__file__).resolve().parent

DATA_PATH = (
    APP_DIR
    / "results"
    / "data"
    / "fairness_summary_table.csv"
)

# ============================================================
# COLUMN NAMES
# ============================================================
MODEL_COL = "Model"
CODEC_COL = "Codec"
LEVEL_COL = "Level"
ETHNIC_COL = "Ethnic"

GENUINE_MEAN_COL = "raw_genuine_mean (%)"
GENUINE_STD_COL = "std_genuine (percentage points)"
IMPOSTOR_MEAN_COL = "raw_impostor_mean (%)"
IMPOSTOR_STD_COL = "std_impostor (percentage points)"

NORM_GENUINE_COL = "norm_genuine_mean"
NORM_IMPOSTOR_COL = "norm_impostor_mean"

FRR_COL = "FRR (%)"
FAR_COL = "FAR (%)"
EER_COL = "EER (%)"
THRESHOLD_COL = "Threshold (normalized score)"

MACRO_EER_COL = "Macro-Average EER (%)"
SER_COL = "SER"


# ============================================================
# DATA LOADING
# ============================================================
@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find the result file at:\n{path}"
        )

    # Auto-detect delimiter to make the app tolerant of CSV
    # exports from different spreadsheet applications.
    df = pd.read_csv(
        path,
        sep=None,
        engine="python",
        skipinitialspace=True,
    )

    # Remove hidden BOM characters / whitespace from headers.
    df.columns = (
        df.columns
        .astype(str)
        .str.replace("\ufeff", "", regex=False)
        .str.replace('"', "", regex=False)
        .str.strip()
    )

    # Older exports may contain a saved DataFrame index.
    for unnecessary_column in [
        "index",
        "Unnamed: 0",
    ]:
        if unnecessary_column in df.columns:
            df = df.drop(
                columns=unnecessary_column
            )

    required_columns = [
        MODEL_COL,
        CODEC_COL,
        LEVEL_COL,
        ETHNIC_COL,
        GENUINE_MEAN_COL,
        GENUINE_STD_COL,
        IMPOSTOR_MEAN_COL,
        IMPOSTOR_STD_COL,
        NORM_GENUINE_COL,
        NORM_IMPOSTOR_COL,
        FRR_COL,
        FAR_COL,
        EER_COL,
        THRESHOLD_COL,
        MACRO_EER_COL,
        SER_COL,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            "The CSV was loaded, but these required columns "
            "were not found:\n\n"
            + "\n".join(
                f"- {column}"
                for column in missing_columns
            )
            + "\n\nDetected columns:\n"
            + "\n".join(
                f"- {column}"
                for column in df.columns
            )
        )

    # Standardize categorical values.
    df[MODEL_COL] = (
        df[MODEL_COL]
        .astype(str)
        .str.strip()
    )

    df[CODEC_COL] = (
        df[CODEC_COL]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df[LEVEL_COL] = (
        df[LEVEL_COL]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    df[ETHNIC_COL] = (
        df[ETHNIC_COL]
        .astype(str)
        .str.strip()
        .str.title()
    )

    numeric_columns = [
        GENUINE_MEAN_COL,
        GENUINE_STD_COL,
        IMPOSTOR_MEAN_COL,
        IMPOSTOR_STD_COL,
        NORM_GENUINE_COL,
        NORM_IMPOSTOR_COL,
        FRR_COL,
        FAR_COL,
        EER_COL,
        THRESHOLD_COL,
        MACRO_EER_COL,
        SER_COL,
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

    return df


# ============================================================
# HELPERS
# ============================================================
def level_number(level: str) -> int:
    try:
        return int(
            str(level)
            .upper()
            .replace("L", "")
        )
    except ValueError:
        return 999


def level_sort(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()

    result["_level_order"] = (
        result[LEVEL_COL]
        .map(level_number)
    )

    result = (
        result
        .sort_values("_level_order")
        .drop(columns="_level_order")
    )

    return result


def codec_display_name(codec: str) -> str:
    mapping = {
        "JPEG": "JPEG",
        "JXL": "JPEG XL",
        "HEIC": "HEIC",
        "NONE": "L0 baseline",
    }

    return mapping.get(
        codec,
        codec,
    )

def model_display_name(model: str) -> str:
    mapping = {
        "arcface": "ArcFace",
        "magface": "MagFace",
        "mobilefacenet": "MobileFaceNet",
    }

    return mapping.get(
        model.lower(),
        model,
    )

def style_chart(
    fig,
    legend_title: str | None = None,
):
    """
    Apply consistent presentation styling to all Plotly charts.
    """

    fig.update_layout(
        margin=dict(
            l=20,
            r=20,
            t=60,
            b=25,
        ),
        title=dict(
            font=dict(
                size=18,
            ),
        ),
        legend=dict(
            title_text=legend_title,
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        hovermode="x unified",
    )

    fig.update_traces(
        connectgaps=False,
        marker=dict(
            size=7,
        ),
    )

    return fig

def add_l0_to_codec(
    df: pd.DataFrame,
    model: str,
    codec: str,
    ethnic: str | None = None,
) -> pd.DataFrame:
    """
    Attach the common L0 baseline to one codec trend.

    L0 is stored as Codec=NONE in the result table.
    """

    compressed = df[
        (df[MODEL_COL] == model)
        & (df[CODEC_COL] == codec)
    ].copy()

    baseline = df[
        (df[MODEL_COL] == model)
        & (df[CODEC_COL] == "NONE")
        & (df[LEVEL_COL] == "L0")
    ].copy()

    if ethnic is not None:
        compressed = compressed[
            compressed[ETHNIC_COL]
            == ethnic
        ].copy()

        baseline = baseline[
            baseline[ETHNIC_COL]
            == ethnic
        ].copy()

    # Duplicate the common baseline only for visualization.
    baseline[CODEC_COL] = codec

    combined = pd.concat(
        [
            baseline,
            compressed,
        ],
        ignore_index=True,
    )

    return level_sort(
        combined
    )


def all_codec_trend(
    df: pd.DataFrame,
    model: str,
    ethnic: str,
) -> pd.DataFrame:
    codecs = [
        codec
        for codec in [
            "JPEG",
            "JXL",
            "HEIC",
        ]
        if codec in set(
            df[CODEC_COL]
        )
    ]

    pieces = [
        add_l0_to_codec(
            df=df,
            model=model,
            codec=codec,
            ethnic=ethnic,
        )
        for codec in codecs
    ]

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    result["Codec Display"] = (
        result[CODEC_COL]
        .map(codec_display_name)
    )

    return result


def macro_eer_trend(
    df: pd.DataFrame,
    model: str,
) -> pd.DataFrame:
    """
    Macro EER is repeated across ethnicity rows in the source
    table. Keep only one row per Model x Codec x Level.
    """

    deduplicated = (
        df[
            df[MODEL_COL]
            == model
        ]
        .drop_duplicates(
            subset=[
                MODEL_COL,
                CODEC_COL,
                LEVEL_COL,
            ]
        )
        .copy()
    )

    baseline = deduplicated[
        (deduplicated[CODEC_COL] == "NONE")
        & (deduplicated[LEVEL_COL] == "L0")
    ].copy()

    pieces = []

    for codec in [
        "JPEG",
        "JXL",
        "HEIC",
    ]:
        codec_rows = deduplicated[
            deduplicated[CODEC_COL]
            == codec
        ].copy()

        if codec_rows.empty:
            continue

        codec_baseline = baseline.copy()
        codec_baseline[CODEC_COL] = codec

        pieces.append(
            pd.concat(
                [
                    codec_baseline,
                    codec_rows,
                ],
                ignore_index=True,
            )
        )

    result = pd.concat(
        pieces,
        ignore_index=True,
    )

    result["Codec Display"] = (
        result[CODEC_COL]
        .map(codec_display_name)
    )

    return level_sort(
        result
    )


# ============================================================
# LOAD DATA
# ============================================================
try:
    df = load_data(
        DATA_PATH
    )

except FileNotFoundError as error:
    st.error(
        str(error)
    )
    st.stop()

except ValueError as error:
    st.error(
        str(error)
    )
    st.stop()


# ============================================================
# SIDEBAR FILTERS
# ============================================================
st.sidebar.header(
    "Explore results"
)

models = sorted(
    df[MODEL_COL]
    .dropna()
    .unique()
)

codecs = [
    codec
    for codec in [
        "JPEG",
        "JXL",
        "HEIC",
    ]
    if codec in set(
        df[CODEC_COL]
    )
]

ethnicities = [
    ethnic
    for ethnic in [
        "Malay",
        "Chinese",
        "Indian",
    ]
    if ethnic in set(
        df[ETHNIC_COL]
    )
]

selected_model = st.sidebar.selectbox(
    "Recognition model",
    models,
    format_func=model_display_name,
)

selected_codec = (
    st.sidebar.selectbox(
        "Codec",
        codecs,
        format_func=codec_display_name,
    )
)

selected_ethnic = (
    st.sidebar.selectbox(
        "Demographic group",
        ethnicities,
    )
)

st.sidebar.caption(
    "L0 is uncompressed baseline."
)


# ============================================================
# TITLE / OVERVIEW
# ============================================================
st.title("Image Compression Effects on Face Recognition Fairness")

st.caption(
    "Responsible AI • Computer Vision • Face Recognition Evaluation"
)

st.markdown(
    """
This interactive dashboard evaluates how **JPEG, JPEG XL, and HEIC**
compression affect face-recognition performance and demographic fairness
across **Malay, Chinese, and Indian** groups.

The study compares **three pretrained face-recognition model conditions**
across compression levels **L0–L5**, using only **aggregate evaluation results**
for privacy-preserving analysis.
"""
)

st.info(
    "Only aggregate evaluation results are displayed. Participant facial images "
    "and participant-level pairwise scores are not included."
)

# ============================================================
# OVERVIEW
# ============================================================

st.subheader("Overview")

with st.container(border=True):

    scope_1, scope_2, scope_3, scope_4 = st.columns(4)

    with scope_1:
        st.metric(
            "Models",
            len(models),
        )

    with scope_2:
        st.metric(
            "Codecs",
            len(codecs),
        )

    with scope_3:
        st.metric(
            "Demographic Groups",
            len(ethnicities),
        )

    with scope_4:
        st.metric(
            "Compression Levels",
            "L0–L5",
        )

# ============================================================
# SELECTED CONDITION SUMMARY
# ============================================================
st.markdown(
    f"""
### Selected L5 Condition
**{model_display_name(selected_model)}** · 
**{codec_display_name(selected_codec)}**
"""
)

st.caption(
    "These metrics summarize the strongest compression condition (L5). "
    "Genuine similarity reflects the selected demographic group, while "
    "Macro-Average EER and SER summarize performance and disparity across all groups."
)

selected_l5 = df[
    (df[MODEL_COL] == selected_model)
    & (df[CODEC_COL] == selected_codec)
    & (df[LEVEL_COL] == "L5")
].copy()


# Selected demographic result
selected_ethnic_l5 = selected_l5[
    selected_l5[ETHNIC_COL] == selected_ethnic
].copy()


if not selected_l5.empty:

    # Genuine similarity follows the selected demographic filter
    genuine_value = (
        selected_ethnic_l5[GENUINE_MEAN_COL].iloc[0]
        if not selected_ethnic_l5.empty
        else np.nan
    )

    # Macro-EER and SER are aggregate fairness metrics,
    # so take one value rather than averaging repeated ethnicity rows.
    macro_values = (
        selected_l5[MACRO_EER_COL]
        .dropna()
    )

    ser_values = (
        selected_l5[SER_COL]
        .dropna()
    )

    macro_eer = (
        macro_values.iloc[0]
        if not macro_values.empty
        else np.nan
    )

    ser = (
        ser_values.iloc[0]
        if not ser_values.empty
        else np.nan
    )


    with st.container(border=True):
        result_1, result_2, result_3 = st.columns(3)

        result_1.metric(
            f"{selected_ethnic} Genuine Similarity",
            (
                f"{genuine_value:.2f}%"
                if pd.notna(genuine_value)
                else "N/A"
            ),
            help=(
                f"Genuine similarity for the selected "
                f"{selected_ethnic} demographic group at L5."
            ),
        )

        result_2.metric(
            "Macro-Average EER",
            (
                f"{macro_eer:.2f}%"
                if pd.notna(macro_eer)
                else "N/A"
            ),
            help=(
                "Average EER across Malay, Chinese, and Indian groups."
            ),
        )

        result_3.metric(
            "SER",
            (
                f"{ser:.2f}"
                if pd.notna(ser)
                else "Undefined"
            ),
            help=(
                "Skewed Error Ratio across demographic groups. "
                "Values closer to 1 indicate more balanced error rates."
            ),
        )

    st.caption(
        "↑ Higher genuine similarity is better • "
        "↓ Lower Macro-Average EER is better • "
        "SER closer to 1 indicates more balanced demographic error rates"
    )

# ============================================================
# COMPRESSION IMPACT
# ============================================================
st.divider()

st.subheader(
    "Compression Impact"
)

st.markdown(
    """
#### Genuine Similarity

**How to interpret:** Genuine similarity measures how similar a compressed
face remains to its corresponding baseline identity. Follow each codec
from **L0 → L5**. A larger downward trend indicates greater recognition
degradation as compression becomes stronger.

**Better result:** higher genuine similarity and a flatter curve.
"""
)

st.caption(
    "Track how recognition similarity changes as compression becomes stronger."
)


recognition_df = all_codec_trend(
    df=df,
    model=selected_model,
    ethnic=selected_ethnic,
)


fig_genuine = px.line(
    recognition_df,
    x=LEVEL_COL,
    y=GENUINE_MEAN_COL,
    color="Codec Display",
    markers=True,
    category_orders={
        LEVEL_COL: [
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
        ]
    },
    labels={
        LEVEL_COL:
            "Compression level",
        GENUINE_MEAN_COL:
            "Genuine similarity (%)",
        "Codec Display":
            "Codec",
    },
    title=(
    f"Genuine Similarity — "
    f"{model_display_name(selected_model)}, {selected_ethnic}"
    ),
)

fig_genuine = style_chart(
    fig_genuine,
    legend_title="Codec",
)

st.plotly_chart(
    fig_genuine,
    use_container_width=True,
)

# ============================================================
# DYNAMIC KEY TAKEAWAY
# ============================================================

l5_comparison = df[
    (df[MODEL_COL] == selected_model)
    & (df[ETHNIC_COL] == selected_ethnic)
    & (df[LEVEL_COL] == "L5")
    & (df[CODEC_COL] != "NONE")
][
    [
        CODEC_COL,
        GENUINE_MEAN_COL,
    ]
].dropna().copy()


if not l5_comparison.empty:

    lowest_row = l5_comparison.loc[
        l5_comparison[GENUINE_MEAN_COL].idxmin()
    ]

    highest_row = l5_comparison.loc[
        l5_comparison[GENUINE_MEAN_COL].idxmax()
    ]

    lowest_codec = codec_display_name(
        lowest_row[CODEC_COL]
    )

    highest_codec = codec_display_name(
        highest_row[CODEC_COL]
    )

    lowest_value = lowest_row[
        GENUINE_MEAN_COL
    ]

    highest_value = highest_row[
        GENUINE_MEAN_COL
    ]

    st.info(
        f"""
**Key takeaway — L5:** For the **{selected_ethnic}** group using
**{model_display_name(selected_model)}**, **{highest_codec}**
preserves the highest genuine similarity at **{highest_value:.2f}%**,
while **{lowest_codec}** produces the lowest at **{lowest_value:.2f}%**.
"""
    )

st.markdown(
        """
    #### Macro-Average EER

    **How to interpret:** Macro-Average EER is the average Equal Error Rate
    across Malay, Chinese, and Indian groups. It provides one overall
    recognition error measure for each codec and compression level.

    **Better result:** lower values. An increase at stronger compression
    levels indicates poorer recognition performance.
    """
)

st.caption(
    "↓ Lower Macro-Average EER indicates better overall recognition performance."
)


macro_df = macro_eer_trend(
    df=df,
    model=selected_model,
)

fig_macro = px.line(
    macro_df,
    x=LEVEL_COL,
    y=MACRO_EER_COL,
    color="Codec Display",
    markers=True,
    category_orders={
        LEVEL_COL: [
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
        ]
    },
    labels={
        LEVEL_COL:
            "Compression level",
        MACRO_EER_COL:
            "Macro-Average EER (%)",
        "Codec Display":
            "Codec",
    },
    title=(
        f"Macro-Average EER — "
        f"{model_display_name(selected_model)}"
    ),
)

fig_macro = style_chart(
    fig_macro,
    legend_title="Codec",
)

st.plotly_chart(
    fig_macro,
    use_container_width=True,
)

# ============================================================
# DYNAMIC MACRO-EER TAKEAWAY
# ============================================================

macro_l5 = (
    df[
        (df[MODEL_COL] == selected_model)
        & (df[LEVEL_COL] == "L5")
        & (df[CODEC_COL] != "NONE")
    ][
        [
            CODEC_COL,
            MACRO_EER_COL,
        ]
    ]
    .drop_duplicates()
    .dropna()
    .copy()
)


if not macro_l5.empty:

    lowest_row = macro_l5.loc[
        macro_l5[MACRO_EER_COL].idxmin()
    ]

    highest_row = macro_l5.loc[
        macro_l5[MACRO_EER_COL].idxmax()
    ]

    lowest_codec = codec_display_name(
        lowest_row[CODEC_COL]
    )

    highest_codec = codec_display_name(
        highest_row[CODEC_COL]
    )

    lowest_value = lowest_row[
        MACRO_EER_COL
    ]

    highest_value = highest_row[
        MACRO_EER_COL
    ]

    st.info(
        f"""
**Key takeaway — L5:** For **{model_display_name(selected_model)}**,
**{lowest_codec}** has the lowest Macro-Average EER at
**{lowest_value:.2f}%**, while **{highest_codec}** has the highest
at **{highest_value:.2f}%**.

A lower Macro-Average EER indicates better overall recognition performance
across the three demographic groups.
"""
    )

# ============================================================
# FAR VS FRR
# ============================================================

st.divider()

st.subheader(
    "Recognition Error Analysis"
)


st.markdown(
    """
#### FAR vs FRR

**How to interpret:** FAR measures how often impostor comparisons are
incorrectly accepted, while FRR measures how often genuine comparisons
are incorrectly rejected.

This chart shows how both error types change as compression becomes
stronger for the selected demographic group.
"""
)

st.caption(
    "FAR ↑ = more false acceptances • "
    "FRR ↑ = more false rejections"
)

error_df = add_l0_to_codec(
    df=df,
    model=selected_model,
    codec=selected_codec,
    ethnic=selected_ethnic,
)

error_long = error_df.melt(
    id_vars=[
        LEVEL_COL,
    ],
    value_vars=[
        FAR_COL,
        FRR_COL,
    ],
    var_name="Error type",
    value_name="Error rate (%)",
)

error_long[
    "Error type"
] = (
    error_long[
        "Error type"
    ]
    .replace(
        {
            FAR_COL: "FAR",
            FRR_COL: "FRR",
        }
    )
)

fig_error = px.line(
    error_long,
    x=LEVEL_COL,
    y="Error rate (%)",
    color="Error type",
    markers=True,
    category_orders={
        LEVEL_COL: [
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
        ]
    },
    labels={
        LEVEL_COL:
            "Compression level",
        "Error rate (%)":
            "Error rate (%)",
    },
    title=(
        "FAR vs FRR"
        f" — {model_display_name(selected_model)}, "
        f"{codec_display_name(selected_codec)}, "
        f"{selected_ethnic}"
    ),
)

fig_error = style_chart(
    fig_error,
    legend_title="Error Type",
)

st.plotly_chart(
    fig_error,
    use_container_width=True,
)

# ============================================================
# DYNAMIC FAR / FRR TAKEAWAY
# ============================================================

far_frr_l5 = df[
    (df[MODEL_COL] == selected_model)
    & (df[CODEC_COL] == selected_codec)
    & (df[ETHNIC_COL] == selected_ethnic)
    & (df[LEVEL_COL] == "L5")
][
    [
        FAR_COL,
        FRR_COL,
        EER_COL,
    ]
].dropna().copy()


if not far_frr_l5.empty:

    far_value = far_frr_l5[FAR_COL].iloc[0]
    frr_value = far_frr_l5[FRR_COL].iloc[0]
    eer_value = far_frr_l5[EER_COL].iloc[0]

    error_gap = abs(far_value - frr_value)

    if far_value > frr_value:
        dominant_error = "false acceptances"
    elif frr_value > far_value:
        dominant_error = "false rejections"
    else:
        dominant_error = "neither error type"

    st.info(
    f"""
    **Key takeaway — L5:** For **{model_display_name(selected_model)}**
    with **{codec_display_name(selected_codec)}** on the
    **{selected_ethnic}** group, FAR is **{far_value:.2f}%** and
    FRR is **{frr_value:.2f}%**.

    The EER is **{eer_value:.2f}%**, with an absolute FAR–FRR gap of
    **{error_gap:.2f} percentage points**. At this operating point,
    **{dominant_error}** is higher.
    """
    )

# ============================================================
# FAIRNESS ANALYSIS
# ============================================================
st.divider()

st.subheader(
    "Fairness Across Demographic Groups"
)

st.markdown(
    """
    #### EER by Demographic Group

    **How to interpret:** Each line represents the EER of one demographic group.
    Compare both the **height** and the **distance between the lines** as
    compression increases.

    Higher EER means more recognition errors, while wider separation between
    groups indicates greater demographic disparity.

    **Better result:** low EER values with demographic lines remaining close together.
    """
)

st.caption(
    "↓ Lower EER is better • "
    "Closer demographic lines indicate lower disparity"
)


fairness_df = add_l0_to_codec(
    df=df,
    model=selected_model,
    codec=selected_codec,
)

fig_eer = px.line(
    fairness_df,
    x=LEVEL_COL,
    y=EER_COL,
    color=ETHNIC_COL,
    markers=True,
    category_orders={
        LEVEL_COL: [
            "L0",
            "L1",
            "L2",
            "L3",
            "L4",
            "L5",
        ],
        ETHNIC_COL: [
            "Malay",
            "Chinese",
            "Indian",
        ],
    },
    labels={
        LEVEL_COL:
            "Compression level",
        EER_COL:
            "EER (%)",
        ETHNIC_COL:
            "Demographic group",
    },
    title=(
        f"EER across Demographic Groups — "
        f" {model_display_name(selected_model)}, "
        f"{codec_display_name(selected_codec)}"
    ),
)

fig_eer = style_chart(
    fig_eer,
    legend_title="Demographic Group",
)

st.plotly_chart(
    fig_eer,
    use_container_width=True,
)

# ============================================================
# DYNAMIC DEMOGRAPHIC EER TAKEAWAY
# ============================================================

eer_l5 = (
    df[
        (df[MODEL_COL] == selected_model)
        & (df[CODEC_COL] == selected_codec)
        & (df[LEVEL_COL] == "L5")
    ][
        [
            ETHNIC_COL,
            EER_COL,
        ]
    ]
    .dropna()
    .copy()
)


if not eer_l5.empty:

    # Round to the same precision shown in the dashboard.
    eer_l5["_display_eer"] = (
        eer_l5[EER_COL]
        .round(2)
    )

    highest_eer = (
        eer_l5["_display_eer"]
        .max()
    )

    lowest_eer = (
        eer_l5["_display_eer"]
        .min()
    )

    eer_gap = (
        highest_eer
        - lowest_eer
    )


    # --------------------------------------------------------
    # Case 1: All demographic groups have the same EER
    # --------------------------------------------------------

    if highest_eer == lowest_eer:

        groups = (
            eer_l5[ETHNIC_COL]
            .tolist()
        )

        group_text = ", ".join(
            groups
        )

        st.info(
            f"""
**Key takeaway — L5:** For **{model_display_name(selected_model)}**
with **{codec_display_name(selected_codec)}**, all evaluated demographic
groups (**{group_text}**) recorded the same EER of **{highest_eer:.2f}%**.

The demographic EER gap at L5 is therefore
**0.00 percentage points**, indicating no EER disparity between the
groups under this specific condition.
"""
        )


    # --------------------------------------------------------
    # Case 2: Demographic EER values are different
    # --------------------------------------------------------

    else:

        highest_groups = (
            eer_l5[
                eer_l5["_display_eer"]
                == highest_eer
            ][ETHNIC_COL]
            .tolist()
        )

        lowest_groups = (
            eer_l5[
                eer_l5["_display_eer"]
                == lowest_eer
            ][ETHNIC_COL]
            .tolist()
        )

        highest_group_text = ", ".join(
            highest_groups
        )

        lowest_group_text = ", ".join(
            lowest_groups
        )

        st.info(
            f"""
**Key takeaway — L5:** For **{model_display_name(selected_model)}**
with **{codec_display_name(selected_codec)}**, the highest EER is observed
for **{highest_group_text}** at **{highest_eer:.2f}%**, while the lowest
is observed for **{lowest_group_text}** at **{lowest_eer:.2f}%**.

The demographic EER gap at L5 is **{eer_gap:.2f} percentage points**.
A larger gap indicates greater disparity in recognition error rates
across demographic groups.
"""
        )

# ============================================================
# SER
# ============================================================
st.markdown(
    """
    #### Skewed Error Ratio (SER)
**How to interpret:** SER compares the **highest demographic EER**
with the **lowest demographic EER** under the same experimental condition.

A SER of **1.00** means the demographic groups have equal EER values.
As SER increases above 1, the disparity between demographic error rates
becomes larger.

**Better result:** values closer to **1.00** indicate more balanced
error rates across demographic groups.
"""
)

st.caption(
    "SER closer to 1 = more balanced demographic error rates"
)


# ------------------------------------------------------------
# Prepare SER data for the selected model
# ------------------------------------------------------------

ser_source = (
    df[
        (df[MODEL_COL] == selected_model)
        & (df[CODEC_COL] != "NONE")
    ][
        [
            MODEL_COL,
            CODEC_COL,
            LEVEL_COL,
            SER_COL,
        ]
    ]
    .drop_duplicates(
        subset=[
            MODEL_COL,
            CODEC_COL,
            LEVEL_COL,
        ]
    )
    .copy()
)


ser_source[
    "Codec Display"
] = (
    ser_source[
        CODEC_COL
    ]
    .map(
        codec_display_name
    )
)


# Rows where SER is mathematically defined
ser_defined = (
    ser_source
    .dropna(
        subset=[
            SER_COL
        ]
    )
    .copy()
)


# ------------------------------------------------------------
# Display chart or explanation
# ------------------------------------------------------------

if ser_defined.empty:

    st.info(
        f"""
SER is undefined for all displayed compression conditions for
**{model_display_name(selected_model)}**.

This occurs when the minimum demographic EER is **0%**, because

**SER = maximum demographic EER / minimum demographic EER**

would require division by zero.

This does **not** indicate missing data.
"""
    )

else:

    fig_ser = px.line(
        level_sort(
            ser_source
        ),
        x=LEVEL_COL,
        y=SER_COL,
        color="Codec Display",
        markers=True,
        category_orders={
            LEVEL_COL: [
                "L1",
                "L2",
                "L3",
                "L4",
                "L5",
            ]
        },
        labels={
            LEVEL_COL:
                "Compression level",
            SER_COL:
                "SER",
            "Codec Display":
                "Codec",
        },
        title=(
            f"SER Where Defined — "
            f"{model_display_name(selected_model)}"
        ),
    )

    fig_ser = style_chart(
        fig_ser,
        legend_title="Codec",
    )

    st.plotly_chart(
        fig_ser,
        use_container_width=True,
    )


    # --------------------------------------------------------
    # Dynamic L5 takeaway
    # --------------------------------------------------------

    ser_l5 = (
        ser_defined[
            ser_defined[
                LEVEL_COL
            ] == "L5"
        ]
        .copy()
    )


    if not ser_l5.empty:

        lowest_row = ser_l5.loc[
            ser_l5[
                SER_COL
            ].idxmin()
        ]

        highest_row = ser_l5.loc[
            ser_l5[
                SER_COL
            ].idxmax()
        ]

        lowest_codec = codec_display_name(
            lowest_row[
                CODEC_COL
            ]
        )

        highest_codec = codec_display_name(
            highest_row[
                CODEC_COL
            ]
        )

        lowest_ser = lowest_row[
            SER_COL
        ]

        highest_ser = highest_row[
            SER_COL
        ]


        if len(ser_l5) == 1:

            st.info(
                f"""
**Key takeaway — L5:** For **{model_display_name(selected_model)}**,
the only defined SER at L5 is for **{lowest_codec}**, with a value of
**{lowest_ser:.2f}**.

SER values closer to **1.00** indicate more balanced demographic
error rates.
"""
            )

        else:

            st.info(
                f"""
**Key takeaway — L5:** For **{model_display_name(selected_model)}**,
**{lowest_codec}** has the lowest defined SER at **{lowest_ser:.2f}**,
while **{highest_codec}** has the highest at **{highest_ser:.2f}**.

Among the defined L5 results, the lower SER indicates more similar
error rates across demographic groups.
"""
            )


# ------------------------------------------------------------
# SER definition note
# ------------------------------------------------------------

st.caption(
    "SER = maximum demographic EER / minimum demographic EER. "
    "SER is undefined when the minimum demographic EER is zero."
)

# ============================================================
# ADVANCED TABLE
# ============================================================
with st.expander("Advanced Metrics & Detailed Results"):

    st.markdown(
        """
        **How to interpret this table**

        - **Raw genuine mean:** similarity between matching identities.
        - **Raw impostor mean:** similarity between different identities.
        - **Normalized means:** scores after impostor Z-normalization.
        - **FAR:** false acceptance rate.
        - **FRR:** false rejection rate.
        - **EER:** operating point where FAR and FRR are approximately balanced.
        - **Threshold:** normalized decision threshold used for the evaluation.

        A larger separation between genuine and impostor scores is generally
        preferable.
        """
    )

    advanced_df = add_l0_to_codec(
        df=df,
        model=selected_model,
        codec=selected_codec,
        ethnic=selected_ethnic,
    )[
        [
            LEVEL_COL,
            GENUINE_MEAN_COL,
            GENUINE_STD_COL,
            IMPOSTOR_MEAN_COL,
            IMPOSTOR_STD_COL,
            NORM_GENUINE_COL,
            NORM_IMPOSTOR_COL,
            FAR_COL,
            FRR_COL,
            EER_COL,
            THRESHOLD_COL,
        ]
    ].copy()

    numeric_columns = (
        advanced_df
        .select_dtypes(
            include="number"
        )
        .columns
    )

    advanced_df[
        numeric_columns
    ] = (
        advanced_df[
            numeric_columns
        ]
        .round(2)
    )

    # ------------------------------------------------------------
    # Rename columns for display only
    # ------------------------------------------------------------

    advanced_display = advanced_df.rename(
        columns={
            LEVEL_COL: "Level",
            GENUINE_MEAN_COL: "Genuine Mean (%)",
            GENUINE_STD_COL: "Genuine Std. Dev.",
            IMPOSTOR_MEAN_COL: "Impostor Mean (%)",
            IMPOSTOR_STD_COL: "Impostor Std. Dev.",
            NORM_GENUINE_COL: "Normalized Genuine Mean",
            NORM_IMPOSTOR_COL: "Normalized Impostor Mean",
            FAR_COL: "FAR (%)",
            FRR_COL: "FRR (%)",
            EER_COL: "EER (%)",
            THRESHOLD_COL: "Decision Threshold",
        }
    )

    st.dataframe(
        advanced_display,
        use_container_width=True,
        hide_index=True,
    )

# ============================================================
# METHODOLOGY
# ============================================================

st.divider()

st.subheader("Methodology")

st.caption(
    "The experimental pipeline evaluates how progressively stronger "
    "image compression affects pretrained face-recognition model embeddings "
    "and demographic error rates."
)


# ------------------------------------------------------------
# High-level methodology
# ------------------------------------------------------------

method_1, method_2, method_3 = st.columns(3)


with method_1:

    st.markdown(
        """
#### 1. Compression

- **L0:** standardized uncompressed baseline
- **L1–L5:** progressively stronger compression
- **Codecs:** JPEG, JPEG XL, HEIC

**Interpretation**

Moving from **L0 → L5** represents increasing compression strength.
Comparing recognition results across these levels shows how robust each
model condition is to image degradation.
"""
    )


with method_2:

    st.markdown(
        """
#### 2. Face Embeddings

Three pretrained recognition conditions were evaluated:

- **ArcFace condition**
- **MobileFaceNet condition**
- **MagFace condition**

Each model receives a standardized facial image and returns a
**face embedding** used for similarity comparison.

No recognition model was trained or fine-tuned in this project.
"""
    )


with method_3:

    st.markdown(
        """
#### 3. Evaluation

The extracted embeddings are evaluated using:

- cosine similarity
- genuine and impostor comparisons
- FAR
- FRR
- EER
- Macro-Average EER
- SER
- demographic comparison

**Interpretation**

Recognition performance is evaluated together with differences in
error rates across the three demographic groups.
"""
    )


# ============================================================
# MODEL DETAILS
# ============================================================

with st.expander(
    "Face-recognition model details"
):

    st.markdown(
        """
### ArcFace condition

**Implementation:** InsightFace `antelopev2`

The experiment loads the **`antelopev2` model pack** through InsightFace
and uses its face-recognition component to extract embeddings.

The model is used only for **inference / embedding extraction**.


### MobileFaceNet condition

**Implementation:** InsightFace `buffalo_sc`

The experiment loads the **`buffalo_sc` model pack** through InsightFace
and uses its face-recognition component to extract embeddings.

This provides the lightweight recognition condition used in the experiment.


### MagFace condition

**Architecture:** `iresnet100` / ResNet-100  
**Embedding dimension:** 512  
**Checkpoint:** `magface_epoch_00025.pth`

The pretrained MagFace backbone is reconstructed using the configuration
from the original experiment and the pretrained checkpoint is loaded for
embedding extraction.

MagFace embeddings are extracted through the PyTorch implementation used
by the original workflow.


> **Important:** All three recognition conditions are pretrained.
> This project evaluates their embeddings under image compression;
> it does not train or fine-tune a new face-recognition model.
"""
    )


# ============================================================
# METRIC DEFINITIONS
# ============================================================

with st.expander(
    "Metric definitions"
):

    st.markdown(
        r"""
### False Acceptance Rate (FAR)

Measures how often impostor comparisons are incorrectly accepted
as genuine.

$$
FAR =
\frac{\mathrm{impostor\ scores\ accepted\ as\ genuine}}
{\mathrm{all\ impostor\ scores}}
$$

**Lower is better.**


### False Rejection Rate (FRR)

Measures how often genuine comparisons are incorrectly rejected.

$$
FRR =
\frac{\mathrm{genuine\ scores\ rejected\ as\ impostor}}
{\mathrm{all\ genuine\ scores}}
$$

**Lower is better.**


### Equal Error Rate (EER)

The experiment selects the operating threshold where FAR and FRR
are approximately balanced.

$$
EER =
\frac{FAR + FRR}{2}
$$

**Lower is better.**


### Macro-Average EER

The arithmetic mean of the demographic-group EER values.

It summarizes overall error performance across the evaluated
demographic groups while giving each group equal weight.

**Lower is better.**


### Skewed Error Ratio (SER)

Compares the highest demographic EER with the lowest demographic EER.

$$
SER =
\frac{\max(EER_g)}
{\min(EER_g)}
$$

Values closer to **1.00** indicate more similar error rates across
demographic groups.

SER is undefined when the minimum demographic EER is zero.
"""
    )

# ============================================================
# LIMITATIONS & PRIVACY
# ============================================================

with st.expander("Limitations & Privacy"):

    st.markdown(
        """
### Study Scope

This study evaluates image-compression effects under a specific
experimental setup:

- **162 facial images**
- **3 demographic groups:** Malay, Chinese, and Indian
- **3 pretrained face-recognition model conditions**
- **3 compression codecs:** JPEG, JPEG XL, and HEIC
- **6 compression conditions:** L0–L5

The findings therefore describe the behaviour observed within this
dataset, model set, and compression configuration. They should not be
interpreted as universal performance claims for all face-recognition
systems or populations.

---

### Privacy Protection

Participant privacy is preserved throughout the public portfolio.

The following are **not included** in this Streamlit application or
public repository:

- participant facial images
- identifying participant information
- participant-level embeddings
- participant-level pairwise similarity scores

The dashboard uses only aggregated results from:

`fairness_summary_table.csv`

---

### Interpretation Considerations

Fairness is evaluated by comparing recognition error rates across the
three demographic groups represented in the study.

Metrics such as **EER, Macro-Average EER, and SER** describe disparity
within the evaluated sample and should be interpreted together with the
dataset size and experimental scope.

SER may be undefined when the minimum demographic EER is **0%**, because
the ratio would require division by zero.
"""
    )


st.divider()

st.caption(
    "Portfolio project: Image Compression Effects on "
    "Face Recognition Fairness"
)