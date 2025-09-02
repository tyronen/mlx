import streamlit as st

video_urls = []

st.title("🎬 Transcribe Live Audio Demonstration")

st.markdown(
    """
This application uses a pretrained large model with significant compute requirements.
Watch the demos below to see the capabilities:
"""
)

for i, video_url in enumerate(video_urls):
    st.subheader(f"Demo {i + 1}")
    st.video(video_url)

st.info(
    "💡 Interested in running this yourself? Check out the source code and deployment guide!"
)
