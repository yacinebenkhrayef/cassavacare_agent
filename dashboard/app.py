import streamlit as st

from api_client import AgentAPIClient, AgentAPIError
from state import init_session_state, reset_session
from components.sidebar import render_sidebar
from components.results_placeholder import render_results
from utils import validate_image_file

st.set_page_config(page_title="CassavaCare Agent", page_icon="🍃", layout="wide")
init_session_state()

st.title("🍃 CassavaCare Agent — Diagnostic Dashboard")
st.caption(
    "Upload a cassava leaf photo to get an AI-assisted diagnosis, "
    "treatment recommendation, and explanation."
)

uploaded_file, location, submit_clicked = render_sidebar()
client = AgentAPIClient()

if submit_clicked:
    valid, error_msg = validate_image_file(uploaded_file)
    if not valid:
        st.sidebar.error(error_msg)
    else:
        reset_session()
        image_bytes = uploaded_file.getvalue()
        st.session_state["uploaded_image_bytes"] = image_bytes
        st.session_state["uploaded_image_name"] = uploaded_file.name
        st.session_state["is_processing"] = True

        status_placeholder = st.empty()
        try:
            job_id, status_url = client.submit_diagnosis(image_bytes, uploaded_file.name, location)
            st.session_state["job_id"] = job_id

            def on_poll(status: str):
                status_placeholder.info(f"Status: {status}…")

            with st.spinner("Running diagnosis pipeline…"):
                job_status = client.wait_for_completion(status_url, on_poll=on_poll)
            status_placeholder.empty()

            if job_status.is_failed:
                st.session_state["job_error"] = job_status.error or "Job failed with no error detail."
            else:
                st.session_state["job_result"] = job_status
        except AgentAPIError as exc:
            st.session_state["job_error"] = str(exc)
        finally:
            st.session_state["is_processing"] = False

if st.session_state.get("job_error"):
    st.error(f"Something went wrong: {st.session_state['job_error']}")

if st.session_state.get("job_result") and st.session_state.get("uploaded_image_bytes"):
    render_results(st.session_state["job_result"], st.session_state["uploaded_image_bytes"])
elif not st.session_state.get("is_processing"):
    st.info("👈 Upload a leaf image, enter a location, and click **Diagnose** to get started.")