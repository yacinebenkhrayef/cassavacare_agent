import streamlit as st
from image_utils import blend_gradcam_overlay
from reasoning_utils import parse_trace

_DECISION_META = {
    "apply": ("✅", "Treatment can be applied now", "success"),
    "defer": ("⏳", "Treatment deferred", "warning"),
    "avoid_aerial": ("🚫", "Aerial spraying avoided — ground application still possible", "warning"),
    "no_action_needed": ("🌿", "No action needed", "success"),
    "pending": ("⏱️", "Decision pending", "info"),
}


def render_results(job_status, original_image_bytes):
    result = job_status.result
    if result is None:
        st.error("Job completed but returned no result data.")
        return

    if result.needs_new_image:
        st.warning(
            "⚠️ Confidence too low to diagnose reliably. "
            "Please upload a clearer photo of the leaf."
        )
        st.image(
            original_image_bytes, caption="Uploaded image", use_container_width=True
        )
        if result.final_report:
            st.write(result.final_report)
        return

    st.subheader("Diagnosis Result")
    label = result.pred_disease_short or result.pred_disease or "Unknown"
    is_healthy = result.is_healthy

    tab_diag, tab_gradcam, tab_reasoning, tab_sources, tab_reco = st.tabs(
        [
            "🩺 Diagnosis",
            "🔥 Grad-CAM",
            "🧠 Agent Reasoning",
            "📚 RAG Sources",
            "✅ Recommendation",
        ]
    )

    with tab_diag:
        st.image(
            original_image_bytes, caption="Uploaded image", use_container_width=True
        )
        st.metric("Predicted class", label)
        if result.confidence is not None:
            st.progress(min(result.confidence, 1.0))
            st.caption(f"Confidence: {result.confidence:.1%}")

    with tab_gradcam:
        if is_healthy:
            st.info("No Grad-CAM for a healthy leaf — nothing to localize.")
        else:
            gradcam_bytes = st.session_state.get("gradcam_bytes")
            gradcam_error = st.session_state.get("gradcam_error")

            if gradcam_error:
                st.warning(f"Grad-CAM unavailable: {gradcam_error}")
            elif gradcam_bytes:
                alpha_pct = st.slider(
                    "Grad-CAM opacity",
                    min_value=0,
                    max_value=100,
                    value=50,
                    step=5,
                    help="0% = original photo only, 100% = full Grad-CAM image.",
                )
                blended = blend_gradcam_overlay(
                    original_image_bytes, gradcam_bytes, alpha_pct / 100.0
                )
                st.image(
                    blended, caption="Grad-CAM overlay", use_container_width=True
                )
                st.caption(
                    "Warmer colors (red/yellow) mark the regions the model relied on most "
                    "for this diagnosis."
                )
            else:
                st.info("Grad-CAM heatmap is still loading…")

    with tab_reasoning:
        if result.trace:
            for step in parse_trace(result.trace):
                with st.expander(f"{step.icon} Étape {step.number} – {step.label}", expanded=False):
                    st.write(step.detail)
        else:
            st.info("No reasoning trace available for this diagnosis.")

    with tab_sources:
        if is_healthy:
            st.info("No treatment sources needed — leaf is healthy.")
        elif result.rag_sources:
            sources = sorted(result.rag_sources, key=lambda s: s.score, reverse=True)
            shown_max = min(len(sources), 5)
            show_all = st.toggle(f"Show all {shown_max} sources", value=False)
            limit = 5 if show_all else 3
            for src in sources[:limit]:
                with st.container(border=True):
                    st.markdown(f"**{src.display_title}**")
                    if src.source:
                        st.caption(f"📁 {src.source}")
                    score_pct = max(0.0, min(1.0, src.score))
                    st.progress(score_pct, text=f"Relevance: {src.score:.2f}")
                    if src.text:
                        preview = src.text if len(src.text) <= 240 else src.text[:240] + "…"
                        st.write(preview)
                        if len(src.text) > 240:
                            with st.expander("Show full excerpt"):
                                st.write(src.text)
        else:
            st.info("No sources were retrieved for this diagnosis.")

    with tab_reco:
        if result.weather_error:
            st.warning(f"⚠️ Weather data unavailable: {result.weather_error}")
        elif result.weather:
            w = result.weather
            cols = st.columns(3)
            cols[0].metric("🌧️ Rain probability", f"{w.get('rain_probability', 0) * 100:.0f}%")
            cols[1].metric("💨 Wind speed", f"{w.get('wind_speed_kmh', 0):.1f} km/h")
            cols[2].metric("⏱️ Forecast window", f"{w.get('forecast_hours', '—')}h")

        if result.decision:
            icon, label, level = _DECISION_META.get(result.decision, ("ℹ️", result.decision, "info"))
            {"success": st.success, "warning": st.warning, "info": st.info}[level](f"{icon} **{label}**")
        if result.decision_reason:
            st.caption(result.decision_reason)
        if result.final_report:
            st.divider()
            st.write(result.final_report)