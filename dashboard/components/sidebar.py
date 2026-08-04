import streamlit as st


def render_sidebar():
    st.sidebar.header("🍃 New Diagnosis")
    uploaded_file = st.sidebar.file_uploader(
        "Upload a cassava leaf image", type=["jpg", "jpeg", "png"]
    )
    location = st.sidebar.text_input(
        "Location (city — required, used for the weather-based treatment decision)",
        key="meta_location",
    )

    can_submit = uploaded_file is not None and bool(location.strip())
    if uploaded_file is not None and not location.strip():
        st.sidebar.caption("⚠️ Location is required by the API.")

    submit = st.sidebar.button("🔍 Diagnose", type="primary", disabled=not can_submit)
    return uploaded_file, location, submit