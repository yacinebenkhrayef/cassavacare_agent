import streamlit as st


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
        st.image(original_image_bytes, caption="Uploaded image", use_container_width=True)
        if result.final_report:
            st.write(result.final_report)
        return

    st.subheader("Diagnosis Result")
    label = result.pred_disease_short or result.pred_disease or "Unknown"
    is_healthy = label.lower() == "healthy"

    tab_diag, tab_gradcam, tab_reasoning, tab_sources, tab_reco = st.tabs(
        ["🩺 Diagnosis", "🔥 Grad-CAM", "🧠 Agent Reasoning", "📚 RAG Sources", "✅ Recommendation"]
    )

    with tab_diag:
        st.image(original_image_bytes, caption="Uploaded image", use_container_width=True)
        st.metric("Predicted class", label)
        if result.confidence is not None:
            st.progress(min(result.confidence, 1.0))
            st.caption(f"Confidence: {result.confidence:.1%}")

    with tab_gradcam:
        if is_healthy:
            st.info("No Grad-CAM for a healthy leaf — nothing to localize.")
        else:
            st.info("Grad-CAM heatmap overlay — implemented in Part 2.")

    with tab_reasoning:
        if result.trace:
            for i, step in enumerate(result.trace, start=1):
                st.write(f"**{i}.** {step}")
            st.caption("Raw trace shown as-is — styled decision-tree view comes in Part 3.")
        else:
            st.info("Agent reasoning trace — implemented in Part 3.")

    with tab_sources:
        if is_healthy:
            st.info("No treatment sources needed — leaf is healthy.")
        elif result.rag_sources:
            st.info(f"{len(result.rag_sources)} source(s) retrieved — formatted cards in Part 3.")
        else:
            st.info("RAG source cards — implemented in Part 3.")

    with tab_reco:
        if result.weather_error:
            st.warning(f"Weather data unavailable: {result.weather_error}")
        if result.decision:
            st.write(f"**Decision:** {result.decision}")
        if result.decision_reason:
            st.caption(result.decision_reason)
        if result.final_report:
            st.write(result.final_report)