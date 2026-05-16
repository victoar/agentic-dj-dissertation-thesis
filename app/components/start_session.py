import streamlit as st


def render(on_submit, status: str = "idle", error_message: str = "") -> None:
    """
    Render the idle / session-start screen.

    status:
      "idle"  — show the text input and submit button
      "error" — show the error message then the input so the user can retry

    Loading is handled by st.spinner() in the caller so the spinner covers
    the whole interaction rather than just this component.
    """
    st.markdown("<div style='height:40px'></div>", unsafe_allow_html=True)

    _, col, _ = st.columns([1, 3, 1])
    with col:
        st.markdown(
            "<h3 style='text-align:center;margin-bottom:4px;'>"
            "What do you want to listen to?"
            "</h3>",
            unsafe_allow_html=True,
        )
        st.markdown(
            "<p style='text-align:center;color:#777;font-size:0.9rem;"
            "margin-bottom:20px;'>"
            "Describe a mood, era, activity, or genre and the agent will find "
            "the perfect opening track."
            "</p>",
            unsafe_allow_html=True,
        )

        if status == "error" and error_message:
            st.error(error_message)

        description = st.text_input(
            label="Session description",
            placeholder="2016 clubbing vibes, calm focus music, late night R&B…",
            label_visibility="collapsed",
        )

        submitted = st.button(
            "Start session",
            type="primary",
            use_container_width=True,
            disabled=(not description.strip()),
        )

        if submitted and description.strip():
            on_submit(description.strip())
