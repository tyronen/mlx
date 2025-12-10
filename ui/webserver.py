import sys
import types

import torch
import streamlit as st

# Streamlit's module watcher walks __path__ and torch.classes raises a RuntimeError.
# Provide a harmless dummy module so the watcher doesn't trip over torch.classes.
if not isinstance(getattr(torch, "classes", None), types.ModuleType):
    _dummy_classes = types.ModuleType("torch.classes")
    _dummy_classes.__path__ = []  # satisfy __path__ lookups
    sys.modules["torch.classes"] = _dummy_classes
    torch.classes = _dummy_classes

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
image_caption_page = st.Page(
    "image_caption_page.py", title="Image captioner", icon=":material/image:"
)
whisper_transcode_page = st.Page(
    "whisper_transcode_page.py",
    title="Transcribe live audio demo",
    icon=":material/speech_to_text:",
)
reddit_summariser_page = st.Page(
    "reddit_summariser_page.py",
    title="Reddit summariser demo",
    icon=":material/summarize:",
)

pg = st.navigation(
    [
        simple_mnist_page,
        hn_predict_page,
        msmarco_search_page,
        complex_mnist_page,
        image_caption_page,
        whisper_transcode_page,
        reddit_summariser_page,
    ]
)
st.set_page_config(
    page_title="Tyrone Nicholas ML/AI demos", page_icon=":material/text_fields_alt:"
)
pg.run()
