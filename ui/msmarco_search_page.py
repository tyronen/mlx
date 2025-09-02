import streamlit as st
import requests
import random
from datasets import load_dataset
import pandas as pd

# Configure Streamlit page
st.set_page_config(page_title="MS MARCO Query Tester", page_icon="🔍", layout="wide")

# FastAPI server configuration
FASTAPI_URL = "http://localhost:8000"


@st.cache_data
def load_msmarco_dataset():
    """Load MS MARCO dataset and cache it"""
    with st.spinner("Loading MS MARCO dataset..."):
        dataset = load_dataset("ms_marco", "v1.1")
        return dataset["train"]


def query_fastapi_server(query, correct_id):
    """Send query to FastAPI server and get results"""
    try:
        response = requests.get(
            f"{FASTAPI_URL}/search",
            params={"query": query, "correct_id": correct_id},
            timeout=10,
        )
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
    except requests.exceptions.RequestException as e:
        st.error(f"Connection error: {str(e)}")
        return None


def format_similarity_score(score):
    """Format similarity score with color coding"""
    if score > 0.8:
        return f"🟢 **{score:.3f}**"
    elif score > 0.6:
        return f"🟡 **{score:.3f}**"
    elif score > 0.4:
        return f"🟠 **{score:.3f}**"
    else:
        return f"🔴 **{score:.3f}**"


def display_document(doc_info, title, is_positive=False):
    """Display a document in a nice format"""
    doc_id, similarity, text = doc_info

    # Choose border color based on whether it's positive or retrieved
    border_color = "#28a745" if is_positive else "#007bff"
    bg_color = "#f8fff8" if is_positive else "#f8f9fa"

    st.markdown(
        f"""
    <div style="
        border: 2px solid {border_color}; 
        border-radius: 10px; 
        padding: 15px; 
        margin: 10px 0;
        background-color: {bg_color};
    ">
        <h4 style="color: {border_color}; margin-top: 0;">{title}</h4>
        <p><strong>Document ID:</strong> <code>{doc_id}</code></p>
        <p><strong>Similarity Score:</strong> {format_similarity_score(similarity)}</p>
        <p><strong>Text:</strong></p>
        <p style="
            background-color: white; 
            padding: 10px; 
            border-radius: 5px; 
            border-left: 4px solid {border_color};
            font-style: italic;
        ">{text}</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def main():
    st.title("🔍 MS MARCO Query Tester")
    st.markdown("Test your two-tower search system against MS MARCO ground truth!")

    # Load dataset
    try:
        dataset = load_msmarco_dataset()
        st.success(f"✅ Loaded MS MARCO dataset with {len(dataset)} queries")
    except Exception as e:
        st.error(f"Failed to load MS MARCO dataset: {str(e)}")
        return

    # Sidebar with controls
    st.sidebar.header("Controls")

    # Manual query input option
    st.sidebar.subheader("Manual Query")
    manual_query = st.sidebar.text_input("Enter your own query:")
    if st.sidebar.button("Search Manual Query") and manual_query:
        st.session_state.current_query = manual_query
        st.session_state.current_item = None
        st.session_state.search_results = None

    # Random query selection
    st.sidebar.subheader("Random Query from MS MARCO")
    if st.sidebar.button("🎲 Find Random Query", type="primary"):
        random_idx = random.randint(0, len(dataset) - 1)
        row = dataset[random_idx]

        st.session_state.current_query = row["query"]
        passages = row["passages"]
        pos_index = passages["is_selected"].index(1)
        st.session_state.positive_text = passages["passage_text"][pos_index]
        st.session_state.current_item = row
        st.session_state.random_idx = random_idx
        st.session_state.search_results = None

    # Display current query
    if hasattr(st.session_state, "current_query"):
        st.header("📝 Current Query")

        if hasattr(st.session_state, "random_idx"):
            st.info(f"**Random Index:** {st.session_state.random_idx}")

        st.markdown(
            f"""
        <div style="
            background-color: #e3f2fd; 
            padding: 20px; 
            border-radius: 10px; 
            border-left: 5px solid #2196f3;
            margin: 15px 0;
        ">
            <h3 style="margin-top: 0; color: #1976d2;">"{st.session_state.current_query}"</h3>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Search button
        if st.button("🔍 Search with FastAPI", type="primary"):
            with st.spinner("Searching..."):
                results = query_fastapi_server(st.session_state.current_query)
                st.session_state.search_results = results

        # Display results
        if (
            hasattr(st.session_state, "search_results")
            and st.session_state.search_results
        ):
            col1, col2 = st.columns(2)

            with col1:
                st.header("🤖 Your Model's Top Result")
                if (
                    st.session_state.search_results
                    and "documents" in st.session_state.search_results
                ):
                    documents = st.session_state.search_results["documents"]
                    if documents:
                        top_doc = documents[0]
                        display_document(top_doc, "Top Retrieved Document")

                        # Show all results in expandable section
                        with st.expander("See all retrieved documents"):
                            for i, doc in enumerate(documents[1:], 2):
                                st.markdown(f"**{i}. Document ID:** `{doc[0]}`")
                                st.markdown(
                                    f"**Similarity:** {format_similarity_score(doc[1])}"
                                )
                                st.markdown(f"**Text:** {doc[2][:200]}...")
                                st.markdown("---")
                    else:
                        st.warning("No documents returned")
                else:
                    st.error("Invalid response format")

            with col2:
                st.header("✅ MS MARCO Ground Truth")
                if (
                    hasattr(st.session_state, "current_item")
                    and st.session_state.current_item
                ):
                    # Find the positive passages
                    passages = st.session_state.current_item["passages"]
                    passage_texts = passages["passage_text"]
                    is_selected = passages["is_selected"]

                    positive_passages = [
                        (f"ground_truth_{i}", 1.0, passage_texts[i])
                        for i, selected in enumerate(is_selected)
                        if selected == 1
                    ]

                    if positive_passages:
                        for i, pos_passage in enumerate(positive_passages):
                            display_document(
                                pos_passage,
                                f"Ground Truth Positive {i + 1}",
                                is_positive=True,
                            )
                    else:
                        st.warning("No positive passages found in ground truth")

                    # Show some negative examples too
                    with st.expander("See negative examples from ground truth"):
                        negative_passages = [
                            (f"negative_{i}", 0.0, passage_texts[i][:200] + "...")
                            for i, selected in enumerate(is_selected)
                            if selected == 0
                        ]

                        for i, neg_passage in enumerate(
                            negative_passages[:3]
                        ):  # Show first 3
                            st.markdown(f"**Negative {i + 1}:** {neg_passage[2]}")
                            st.markdown("---")
                else:
                    st.info("Use 'Find Random Query' to see ground truth")

        # Performance analysis
        if (
            hasattr(st.session_state, "search_results")
            and st.session_state.search_results
            and hasattr(st.session_state, "current_item")
            and st.session_state.current_item
        ):

            st.header("📊 Performance Analysis")

            # Check if top result matches any positive passage
            documents = st.session_state.search_results["documents"]
            passages = st.session_state.current_item["passages"]
            passage_texts = passages["passage_text"]
            is_selected = passages["is_selected"]

            positive_texts = [
                passage_texts[i]
                for i, selected in enumerate(is_selected)
                if selected == 1
            ]
            top_retrieved_text = documents[0][2] if documents else ""

            # Simple text matching (you could make this more sophisticated)
            matches = any(
                pos_text.strip() == top_retrieved_text.strip()
                for pos_text in positive_texts
            )

            col1, col2, col3 = st.columns(3)

            with col1:
                if matches:
                    st.success("🎯 **Perfect Match!**\nTop result matches ground truth")
                else:
                    st.error("❌ **No Match**\nTop result doesn't match ground truth")

            with col2:
                if documents:
                    similarity = documents[0][1]
                    st.metric("Top Similarity Score", f"{similarity:.3f}")

            with col3:
                if hasattr(st.session_state.search_results, "get"):
                    latency = st.session_state.search_results.get("Latency", "N/A")
                    st.metric(
                        "Search Latency", f"{latency}ms" if latency != "N/A" else "N/A"
                    )

    else:
        st.info("👈 Click 'Find Random Query' in the sidebar to get started!")

        # Show some dataset statistics
        st.header("📈 Dataset Information")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Total Queries", f"{len(dataset):,}")

        with col2:
            # Sample a few items to get average passage count
            sample_items = [dataset[i] for i in range(0, min(100, len(dataset)), 10)]
            avg_passages = sum(
                len(item["passages"]["passage_text"]) for item in sample_items
            ) / len(sample_items)
            st.metric("Avg Passages per Query", f"{avg_passages:.1f}")

        with col3:
            st.metric(
                "FastAPI Status", "🟢 Ready" if True else "🔴 Down"
            )  # You could add a health check here


if __name__ == "__main__":
    main()
