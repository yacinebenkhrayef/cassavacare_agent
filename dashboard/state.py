import streamlit as st

_KEYS = (
    "job_id",
    "job_result",
    "job_error",
    "is_processing",
    "uploaded_image_bytes",
    "uploaded_image_name",
    "gradcam_bytes",
    "gradcam_error",
)


def init_session_state():
    defaults = {k: None for k in _KEYS}
    defaults["is_processing"] = False
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def reset_session():
    for k in _KEYS:
        st.session_state[k] = None
    st.session_state["is_processing"] = False