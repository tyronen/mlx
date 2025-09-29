import streamlit as st

simple_mnist_page = st.Page(
    "simple_mnist_page.py",
    title="Simple handwriting recogniser",
    icon=":material/edit:",
)
hn_predict_page = st.Page(
    "hn_predict_page.py",
    title="Hacker News prediction",
    icon=":material/batch_prediction:",
)
msmarco_search_page = st.Page(
    "msmarco_search_page.py", title="MS Marco search", icon=":material/search:"
)
complex_mnist_page = st.Page(
    "complex_mnist_page.py",
    title="Complex handwriting recogniser",
    icon=":material/draw:",
)
whisper_transcode_page = st.Page(
    "whisper_transcode_page.py",
    title="Transcribe live audio demo",
    icon=":material/speech_to_text:",
)
image_caption_page = st.Page(
    "image_caption_page.py", title="Image captioner", icon=":material/image:"
)
reddit_summariser_page = st.Page(
    "reddit_summariser_page.py",
    title="Reddit summariser demo",
    icon=":material/summarize:",
)

pg = st.navigation(
    [
        hn_predict_page,
        simple_mnist_page,
        msmarco_search_page,
        complex_mnist_page,
        whisper_transcode_page,
        image_caption_page,
        reddit_summariser_page,
    ]
)
st.set_page_config(
    page_title="Tyrone Nicholas ML/AI demos", page_icon=":material/text_fields_alt:"
)
pg.run()
