import html
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

VERDICT_COLORS = {
    "TRUSTED": "#0ca30c",
    "REVIEW": "#fab219",
    "BLOCKED": "#d03b3b",
}
SEQUENTIAL_BLUE = "#2a78d6"

st.set_page_config(page_title="TrustLens", layout="wide")

st.title("🔍 TrustLens")
st.caption("Scores AI answers for trustworthiness by checking how well they are grounded in their source text.")

STOPWORDS = {
    "a", "an", "the", "and", "or", "but", "if", "then", "than", "so",
    "is", "are", "was", "were", "be", "been", "being",
    "am", "do", "does", "did", "doing",
    "have", "has", "had", "having",
    "i", "you", "he", "she", "it", "we", "they",
    "me", "him", "her", "us", "them",
    "my", "your", "his", "its", "our", "their",
    "this", "that", "these", "those",
    "to", "of", "in", "on", "at", "by", "for", "with", "about",
    "against", "between", "into", "through", "during", "before",
    "after", "above", "below", "from", "up", "down", "out", "off",
    "over", "under", "again", "further", "once",
    "as", "not", "no", "nor",
    "can", "will", "just", "should", "now",
    "there", "here", "what", "which", "who", "whom",
    "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "only", "own", "same",
}


def tokenize(text: str) -> set[str]:
    words = re.findall(r"[a-z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS}


def grounding_score(ai_answer: str, source: str) -> int:
    answer_words = tokenize(ai_answer)
    if not answer_words:
        return 0
    source_words = tokenize(source)
    supported = sum(1 for w in answer_words if w in source_words)
    return round(100 * supported / len(answer_words))


def verdict(score: int) -> str:
    if score >= 75:
        return "TRUSTED"
    if score >= 45:
        return "REVIEW"
    return "BLOCKED"


def render_html(block: str) -> None:
    """Render a triple-quoted HTML block, stripping per-line indentation so
    Streamlit's markdown parser doesn't mistake it for an indented code block."""
    flattened = "\n".join(line.strip() for line in block.strip("\n").splitlines())
    st.markdown(flattened, unsafe_allow_html=True)


def highlight_fabrications(ai_answer: str, source: str) -> str:
    source_words = tokenize(source)
    escaped = html.escape(ai_answer)

    def repl(match: re.Match) -> str:
        word = match.group(0)
        if word.lower() not in STOPWORDS and word.lower() not in source_words:
            return (
                '<mark style="background:#d03b3b; color:#fff; padding:0 4px; '
                'border-radius:3px; font-weight:600;">' + word + "</mark>"
            )
        return word

    return re.sub(r"[A-Za-z0-9]+", repl, escaped)


df = pd.read_csv("ai_test_cases.csv")

df["grounding_score"] = df.apply(
    lambda row: grounding_score(row["ai_answer"], row["source"]), axis=1
)
df["verdict"] = df["grounding_score"].apply(verdict)

st.sidebar.header("Filters")
category_options = sorted(df["category"].unique())
verdict_options = ["TRUSTED", "REVIEW", "BLOCKED"]
selected_categories = st.sidebar.multiselect(
    "Category", category_options, default=category_options
)
selected_verdicts = st.sidebar.multiselect(
    "Verdict", verdict_options, default=verdict_options
)

filtered_df = df[
    df["category"].isin(selected_categories) & df["verdict"].isin(selected_verdicts)
]

if filtered_df.empty:
    st.warning("No test cases match the selected filters.")
    st.stop()

total_cases = len(filtered_df)
avg_score = filtered_df["grounding_score"].mean()
trusted_pct = (filtered_df["verdict"] == "TRUSTED").mean() * 100
hallucination_rate = (filtered_df["verdict"] == "BLOCKED").mean() * 100

col1, col2, col3, col4 = st.columns([1, 1, 1, 1.4])
col1.metric("Total Test Cases", total_cases)
col2.metric("Average Grounding Score", f"{avg_score:.0f}")
col3.metric("% TRUSTED", f"{trusted_pct:.0f}%")
with col4:
    render_html(
        f"""
        <div style="border:2px solid #ff4b4b; border-radius:8px; padding:0.6rem 1rem;">
            <div style="font-size:0.9rem; color:#ff4b4b; font-weight:600;">
                ⚠️ Hallucination Rate
            </div>
            <div style="font-size:2.2rem; font-weight:700; color:#ff4b4b; line-height:1.2;">
                {hallucination_rate:.0f}%
            </div>
            <div style="font-size:0.75rem; color:#888;">% of answers BLOCKED</div>
        </div>
        """
    )

st.divider()

render_html(
    """
    <div style="background:#d03b3b; padding:0.75rem 1.25rem; border-radius:8px 8px 0 0;">
        <span style="color:#fff; font-size:1.4rem; font-weight:700;">🚨 Flagged Answers</span>
        <span style="color:#ffe5e5; font-size:0.9rem;"> — likely fabrications, highlighted below</span>
    </div>
    """
)

flagged_df = filtered_df[filtered_df["verdict"] == "BLOCKED"]

if flagged_df.empty:
    render_html(
        """
        <div style="border:1px solid #e1e0d9; border-top:none; border-radius:0 0 8px 8px;
                    padding:1rem 1.25rem;">
            ✅ No BLOCKED cases in the current filter selection.
        </div>
        """
    )
else:
    cards = []
    for _, case in flagged_df.iterrows():
        highlighted_answer = highlight_fabrications(case["ai_answer"], case["source"])
        cards.append(
            f"""
            <div style="border-left:5px solid #d03b3b; background:rgba(208,59,59,0.06);
                        padding:0.9rem 1.25rem; margin-top:0.75rem; border-radius:0 6px 6px 0;">
                <div style="font-size:0.75rem; font-weight:700; color:#d03b3b; letter-spacing:0.03em;">
                    {html.escape(case['id'])} · {html.escape(case['category'])} · SCORE {case['grounding_score']}
                </div>
                <div style="font-weight:600; margin-top:0.35rem;">{html.escape(case['question'])}</div>
                <div style="margin-top:0.4rem; line-height:1.5;">{highlighted_answer}</div>
            </div>
            """
        )
    render_html(
        f"""
        <div style="border:1px solid #e1e0d9; border-top:none; border-radius:0 0 8px 8px;
                    padding:0.5rem 1.25rem 1.1rem;">
            {''.join(cards)}
        </div>
        """
    )

st.divider()

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Verdict Split")
    verdict_counts = (
        filtered_df["verdict"]
        .value_counts()
        .reindex(["TRUSTED", "REVIEW", "BLOCKED"])
        .fillna(0)
    )
    donut = go.Figure(
        data=[
            go.Pie(
                labels=verdict_counts.index,
                values=verdict_counts.values,
                hole=0.6,
                marker=dict(
                    colors=[VERDICT_COLORS[v] for v in verdict_counts.index],
                    line=dict(color="#fcfcfb", width=2),
                ),
                textinfo="label+percent",
                hovertemplate="%{label}: %{value} cases (%{percent})<extra></extra>",
            )
        ]
    )
    donut.update_layout(
        showlegend=True,
        margin=dict(t=20, b=20, l=20, r=20),
        height=380,
    )
    st.plotly_chart(donut, use_container_width=True)

with chart_col2:
    st.subheader("Avg Grounding Score by Category")
    by_category = (
        filtered_df.groupby("category")["grounding_score"]
        .mean()
        .sort_values()
        .reset_index()
    )
    bar = px.bar(
        by_category,
        x="grounding_score",
        y="category",
        orientation="h",
        color_discrete_sequence=[SEQUENTIAL_BLUE],
        text=by_category["grounding_score"].round(0).astype(int),
    )
    bar.update_traces(
        textposition="outside",
        hovertemplate="%{y}: %{x:.0f}<extra></extra>",
    )
    bar.update_layout(
        margin=dict(t=20, b=20, l=20, r=20),
        height=380,
        xaxis_title="Average grounding score",
        yaxis_title="",
        xaxis_range=[0, 105],
        showlegend=False,
    )
    st.plotly_chart(bar, use_container_width=True)

st.subheader("Grounding Score Distribution")
hist = px.histogram(
    filtered_df,
    x="grounding_score",
    nbins=10,
    color_discrete_sequence=[SEQUENTIAL_BLUE],
)
hist.update_traces(hovertemplate="Score range: %{x}<br>Count: %{y}<extra></extra>")
hist.update_layout(
    margin=dict(t=20, b=20, l=20, r=20),
    height=320,
    xaxis_title="Grounding score",
    yaxis_title="Number of test cases",
    bargap=0.05,
)
st.plotly_chart(hist, use_container_width=True)

st.divider()

st.subheader("Test Cases")
st.dataframe(
    filtered_df[
        ["id", "category", "question", "ai_answer", "grounding_score", "verdict"]
    ],
    use_container_width=True,
)

st.divider()

with st.expander("🔎 Inspect a test case"):
    selected_id = st.selectbox("Test case ID", filtered_df["id"])
    case = filtered_df.loc[filtered_df["id"] == selected_id].iloc[0]

    st.markdown(f"**Category:** {case['category']}")
    st.markdown(f"**Question:** {case['question']}")

    detail_col1, detail_col2 = st.columns(2)
    with detail_col1:
        st.markdown("**Source**")
        st.info(case["source"])
        st.markdown("**AI Answer**")
        st.warning(case["ai_answer"])
    with detail_col2:
        st.markdown("**Expected Answer**")
        st.success(case["expected_answer"])
        st.markdown("**Grounding Score / Verdict**")
        badge_color = VERDICT_COLORS[case["verdict"]]
        render_html(
            f"""
            <div style="display:flex; align-items:center; gap:0.75rem;">
                <span style="font-size:2rem; font-weight:700;">{case['grounding_score']}</span>
                <span style="background:{badge_color}; color:white; padding:0.25rem 0.75rem;
                              border-radius:999px; font-weight:600; font-size:0.85rem;">
                    {case['verdict']}
                </span>
            </div>
            """
        )
