import streamlit as st
from setup import BACKGROUND_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, BLACK_COLOR


def recommendations_app():
    """
    streamlit implementation of the recs page
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

    st.session_state["recs_generated"] = False
    st.title("Your Game Recommendations")
    st.markdown("Click the button below to generate your personalized recommendations.")

    if "username" not in st.session_state or st.session_state["username"] == "":
        st.warning("Please log in to generate recommendations.")
    else:
        if "recommendations" not in st.session_state:
            st.session_state["recommendations"] = []

        if st.button("Generate Recommendations"):
            username = st.session_state["username"]
            api = st.session_state["api"]
            st.session_state["recommendations"] = api.get_recommendations(username, top_n=5, k=10)
            st.session_state["recs_generated"] = True

        # Always display them if they exist:
        recs = st.session_state["recommendations"]
        if recs:
            st.subheader("Recommended Games:")

            # For each recommended game, create a button to jump to the Game Info page.
            # We store the BGGId in session_state so the next page can retrieve it.
            for i, game in enumerate(recs):
                game_id = game["BGGId"]
                game_name = game.get("Name", "Unnamed Game")

                # Create a unique key for each button so they don't collide
                if st.button(f"View details: {game_name}", key=f"rec_button_{i}"):
                    st.session_state["selected_game_id"] = game_id
                    st.switch_page("pages/game_page.py")

        else:
            if st.session_state.get("recs_generated"):
                st.info("No recommendations available at this time. Please try again later.")
            else:
                st.info("Try generating some game recommendations!")


if __name__ == "__main__":
    recommendations_app()
