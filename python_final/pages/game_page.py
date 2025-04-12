import streamlit as st
import json
from api import GameAPI
from setup import GAME_COLUMNS_JSON_PATH, BACKGROUND_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, BLACK_COLOR

# st.set_page_config(page_title="Game Info")

def game_info_page():
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

    # Instead of reading from st.query_params, we retrieve from st.session_state
    bggid = st.session_state.get("selected_game_id", None)
    if bggid is None:
        st.error("No game selected. Please use the Recommendations page or another link to pick a game.")
        return

    # Retrieve GameAPI instance from session state, or create one if needed.
    api = st.session_state.get("api")
    if not api:
        from api import GameAPI
        api = GameAPI()
        st.session_state["api"] = api

    # Suppose your API has a method get_game_info(bggid)
    game_info = api.get_game_info(bggid)
    if game_info is None:
        st.error("Game not found.")
        return

    # Display the game's title and description.
    st.header(game_info.get("Name", "Unknown Game"))
    description = game_info.get("Description", "No description available.")
    st.subheader("Description")
    st.markdown(description)

    # Display game feature flags by category.
    st.subheader("Game Features")
    try:
        with open(GAME_COLUMNS_JSON_PATH, "r") as f:
            game_columns = json.load(f)
    except Exception as e:
        st.error("Error loading game features configuration.")
        return

    for category, col_names in game_columns.items():
        applicable = []
        for col in col_names:
            val = game_info.get(col, 0)
            if str(val) == "1" or val == 1:
                applicable.append(col)
        if applicable:
            st.markdown(f"**{category}**:")
            for tag in applicable:
                st.markdown(f"- {tag}")

    st.markdown("---")
    if st.button("Return to Recommendations"):
        st.switch_page("pages/recs_page.py")


if __name__ == "__main__":
    game_info_page()
