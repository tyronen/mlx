import os
import streamlit as st
import requests
import datetime
# Import the new helper functions
from utils.hn_api import get_item, fetch_random_recent_story

# --- Configuration ---
FASTAPI_URL = os.getenv("FASTAPI_URL", "http://127.0.0.1:8000")

# --- Initialize Session State ---
if 'id' not in st.session_state:
    st.session_state.id = 40646061
if 'by' not in st.session_state:
    st.session_state.by = "testuser"
if 'title' not in st.session_state:
    st.session_state.title = "My awesome new project"
if 'url' not in st.session_state:
    st.session_state.url = "http://example.com"
if 'time_obj' not in st.session_state:
    st.session_state.time_obj = datetime.datetime.now()
if 'score' not in st.session_state:
    st.session_state.score = None
if 'comments' not in st.session_state:
    st.session_state.comments = None

# --- Streamlit UI ---

st.title("Hacker News Post Scorer")

# --- Section 1: Fetching Data ---
st.header("1. Fetch Post Data")
st.write("Enter an ID or fetch a random recent post to populate the fields below.")

# --- Fetch by ID ---
st.subheader("Fetch by ID")
col1, col2 = st.columns([3, 2])
with col1:
    item_id_input = st.number_input("Hacker News Post ID", min_value=1, value=st.session_state['id'], step=1, label_visibility="collapsed")
with col2:
    if st.button("Fetch and Populate"):
        with st.spinner(f"Fetching data for item {item_id_input}..."):
            item_data = get_item(item_id_input)
            if item_data and item_data.get("type") == "story":
                st.session_state.by = item_data.get("by", "")
                st.session_state.title = item_data.get("title", "")
                st.session_state.url = item_data.get("url", "")
                st.session_state.time_obj = datetime.datetime.fromtimestamp(item_data.get("time", 0))
                st.session_state.score = item_data.get("score", 0)
                st.session_state.comments = item_data.get("descendants", 0)
                st.success(f"Successfully populated form with data for ID {item_id_input}.")
            else:
                st.warning(f"Item {item_id_input} is not a story or was not found.")
                st.session_state.score, st.session_state.comments = None, None

# --- Fetch Random ---
st.subheader("Fetch Random Recent Post")
if st.button("Fetch and Populate Random"):
    with st.spinner("Searching for a random recent story..."):
        item_data = fetch_random_recent_story()
        if item_data:
            st.session_state.id = item_data.get("id")
            st.session_state.by = item_data.get("by", "")
            st.session_state.title = item_data.get("title", "")
            st.session_state.url = item_data.get("url", "")
            st.session_state.time_obj = datetime.datetime.fromtimestamp(item_data.get("time", 0))
            st.session_state.score = item_data.get("score", 0)
            st.session_state.comments = item_data.get("descendants", 0)
            st.success(f"Successfully populated form with data for random post ID {item_data.get('id')}.")
        else:
            st.error("Could not find a random recent story. Please try again.")
            st.session_state.score, st.session_state.comments = None, None

# Display metrics if they exist in the session state
if st.session_state.score is not None:
    st.write("---")
    metric_col1, metric_col2 = st.columns(2)
    with metric_col1:
        st.metric("Actual Score", st.session_state.score)
    with metric_col2:
        st.metric("Comments", st.session_state.comments)

st.divider()

# --- Section 2: Prediction ---
st.header("2. Review Data and Predict Score")
st.write("Review or edit the populated data, then click predict.")

by_input = st.text_input("Author (by)", key="by")
title_input = st.text_input("Post Title", key="title")
url_input = st.text_input("URL", key="url")

date_input = st.date_input("Post Date", st.session_state.time_obj.date())
time_input = st.time_input("Post Time", st.session_state.time_obj.time())

datetime_obj = datetime.datetime.combine(date_input, time_input)
time_stamp = int(datetime_obj.timestamp())

st.write(f"_Final Unix Timestamp for API: `{time_stamp}`_")

if st.button("Predict Score"):
    payload = {
        "by": by_input,
        "title": title_input,
        "url": url_input,
        "time": time_stamp
    }
    try:
        with st.spinner("Getting prediction..."):
            response = requests.post(f"{FASTAPI_URL}/predict/direct", json=payload)
            response.raise_for_status()
            prediction_data = response.json()
            st.success("Prediction successful!")
            st.write(f"**Predicted Score:** `{prediction_data['prediction']}`")

            with st.expander("See Raw JSON Response"):
                st.json(prediction_data)

    except requests.exceptions.RequestException as e:
        st.error(f"An error occurred while connecting to the API: {e}")
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
