import streamlit as st
from setup import BACKGROUND_COLOR, PRIMARY_COLOR, BLACK_COLOR, TERNARY_COLOR, SECONDARY_COLOR
from api import GameAPI


def landing_app():
    """
    streamlit implementation of the landing page
    :return: None
    """
    st.markdown(
        f"""
        <style>
        .reportview-container {{
            background-color: {BACKGROUND_COLOR};
            color: {BLACK_COLOR};
        }}
        .sidebar .sidebar-content {{
            background-color: {BACKGROUND_COLOR};
        }}
        .stTextInput>div>input {{
            background-color: {SECONDARY_COLOR};
            color: {BLACK_COLOR};
        }}
        div.stButton > button {{
            background-color: {PRIMARY_COLOR};
            color: {BLACK_COLOR};
            border: none;
            padding: 0.5em 1em;
            border-radius: 5px;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Container for user status display.
    auth_container = st.container()
    with auth_container:
        if st.session_state.get("username", "") == "":
            st.markdown("**Please log in to access personalized recommendations and dashboards.**")

            if st.button("Go to Login Page"):
                st.session_state["current_page"] = "login"
                st.switch_page("pages/login_page.py")
        else:
            col_content, col_auth = st.columns([3, 1])
            with col_auth:
                st.markdown(f"**Logged in as:** {st.session_state['username']}")
                if st.button("Logout"):
                    st.session_state["username"] = ""
                    # After logout, jump to the login page
                    st.switch_page("pages/login_page.py")

    # Main landing page content.
    st.title("Welcome to the Board Game Recommender")
    st.markdown(
        """
        ## Overview
        The Board Game Recommender project leverages deep learning and content-based filtering techniques 
        to help enthusiasts discover new games that match their tastes. By compressing rich game features 
        into a dense latent space with an autoencoder, we can compute meaningful similarities between games 
        and provide personalized recommendations based on your ratings.

        ### What Problem Do We Solve?
        Many board game lovers face the challenge of sifting through an overwhelming number of titles to find games 
        that match their specific interests. Our system automates this process, delivering curated recommendations 
        that keep your unique taste in mind.

        ### Key Features
        - **Autoencoder-based Feature Extraction:** Learn compact representations of games.
        - **Content-Based Filtering:** Recommend games similar to those you’ve loved.
        - **Interactive Visualizations:** Explore trends and data insights across a rich dataset.
        - **User-Centric Design:** A sleek interface that adapts to your login status and personal preferences.
        """
    )
    st.markdown(
        """
        **Note:** Log in to see personalized recommendations and extra dashboards.
        """
    )

if __name__ == "__main__":
    landing_app()
