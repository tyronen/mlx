import datetime

import streamlit as st

# Import the new helper functions
from ui.hn_predict.hn_api import get_item, fetch_random_recent_story
from ui.hn_predict.predict import predict_direct, HNPostData

# --- Initialize Session State ---
if "id" not in st.session_state:
    st.session_state.id = 40646061
if "by" not in st.session_state:
    st.session_state.by = "testuser"
if "title" not in st.session_state:
    st.session_state.title = "My awesome new project"
if "url" not in st.session_state:
    st.session_state.url = "http://example.com"
if "time_obj" not in st.session_state:
    st.session_state.time_obj = datetime.datetime.now()
if "score" not in st.session_state:
    st.session_state.score = None
if "comments" not in st.session_state:
    st.session_state.comments = None

# --- Streamlit UI ---

# Hide spinner arrows on number inputs
st.markdown(
    """
<style>
    /* Hide number input step buttons */
    button[data-testid="stNumberInputStepDown"],
    button[data-testid="stNumberInputStepUp"] {
        display: none !important;
    }
</style>
""",
    unsafe_allow_html=True,
)

st.title("Hacker News Post Scorer")

# --- Section 1: Fetching Data ---
st.write("Fetch a random recent post or enter an ID to populate the fields below.")

random_col, id_col = st.columns(2)
with random_col:
    st.subheader("Random Recent Post")
    if st.button("Fetch and Populate Random"):
        with st.spinner("Searching for a random recent story..."):
            item_data = fetch_random_recent_story()
            if item_data:
                st.session_state.id = item_data.get("id")
                st.session_state.by = item_data.get("by", "")
                st.session_state.title = item_data.get("title", "")
                st.session_state.url = item_data.get("url", "")
                st.session_state.time_obj = datetime.datetime.fromtimestamp(
                    item_data.get("time", 0)
                )
                st.session_state.score = item_data.get("score", 0)
                st.session_state.comments = item_data.get("descendants", 0)
                st.success(
                    f"Successfully populated form with data for random post ID {item_data.get('id')}."
                )
            else:
                st.error("Could not find a random recent story. Please try again.")
                st.session_state.score, st.session_state.comments = None, None

# --- Fetch by ID ---
with id_col:
    st.subheader("By ID")
    text_col, button_col = st.columns(2)
    with text_col:
        item_id_input = st.number_input(
            "Hacker News Post ID",
            min_value=1,
            value=st.session_state["id"],
            step=1,
            label_visibility="collapsed",
        )
    with button_col:
        if st.button("Fetch and Populate"):
            with st.spinner(f"Fetching data for item {item_id_input}..."):
                item_data = get_item(item_id_input)
                if item_data and item_data.get("type") == "story":
                    st.session_state.by = item_data.get("by", "")
                    st.session_state.title = item_data.get("title", "")
                    st.session_state.url = item_data.get("url", "")
                    st.session_state.time_obj = datetime.datetime.fromtimestamp(
                        item_data.get("time", 0)
                    )
                    st.session_state.score = item_data.get("score", 0)
                    st.session_state.comments = item_data.get("descendants", 0)
                    st.success(
                        f"Successfully populated form with data for ID {item_id_input}."
                    )
                else:
                    st.warning(f"Item {item_id_input} is not a story or was not found.")
                    st.session_state.score, st.session_state.comments = None, None


# --- Section 2: Prediction ---
st.markdown(
    f"""
<table>
<tr><td><strong>Author</strong></td><td>{st.session_state.by}</td></tr>
<tr><td><strong>Title</strong></td><td>{st.session_state.title}</td></tr>
<tr><td><strong>URL</strong></td><td>{st.session_state.url}</td></tr>
<tr><td><strong>Date</strong></td><td>{st.session_state.time_obj.date()}</td></tr>
<tr><td><strong>Time</strong></td><td>{st.session_state.time_obj.time()}</td></tr>
</table>
""",
    unsafe_allow_html=True,
)


st.divider()

datetime_obj = datetime.datetime.combine(
    st.session_state.time_obj.date(), st.session_state.time_obj.time()
)
time_stamp = int(datetime_obj.timestamp())
payload = HNPostData(
    by=st.session_state.by,
    title=st.session_state.title,
    url=st.session_state.url,
    time=time_stamp,
)
try:
    with st.spinner("Getting prediction..."):
        prediction = predict_direct(payload)
        prediction_col, actual_col = st.columns(2)
        with prediction_col:
            st.metric("Median Prediction", format(prediction["median"], ".0f"))

            st.caption(
                "Since HN scores are heavily skewed (46% of posts get score ≤1, and 20% get score 2), "
                "we show the full probability distribution."
            )

            # Display quantiles with meaningful labels
            # Note: These percentiles align with natural score boundaries in HN data
            quantile_labels = [
                "46th percentile (~1 point)",
                "66th percentile (~2 points)",
                "75th percentile (~3 points)",
                "90th percentile (~16 points)",
                "97th percentile (~100 points)",
            ]
            quantile_explanations = [
                "46% of similar posts score at or below this value",
                "66% of similar posts score at or below this value",
                "75% of similar posts score at or below this value",
                "90% of similar posts score at or below this value",
                "97% of similar posts score at or below this value",
            ]

            for q, (label, explanation, value) in enumerate(
                zip(quantile_labels, quantile_explanations, prediction["quantiles"])
            ):
                st.metric(label, format(value, ".1f"), help=explanation)

            # Show confidence interval
            if len(prediction["quantiles"]) >= 2:
                low_quantile = prediction["quantiles"][0]  # 46th percentile
                high_quantile = prediction["quantiles"][-1]  # 97th percentile
                st.info(
                    f"**Confidence Range**: {format(low_quantile, '.0f')} - {format(high_quantile, '.1f')} points (46th to 97th percentile)"
                )
        with actual_col:
            if st.session_state.score is not None:
                st.metric("Actual Score", st.session_state.score)
except Exception as e:
    st.error(f"An unexpected error occurred: {e}")
