from setup import BACKGROUND_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, TERNARY_COLOR, BLACK_COLOR
import streamlit as st

# Optional: Set overall configuration for your app
st.set_page_config(
    page_title="Board Game Recommender",
    page_icon=":game_die:",  # or any emoji
    layout="centered"
)

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


# The main page content (before going into any subpages)
st.title("Board Game Recommender App")
st.markdown(
    """
    Welcome! Use the sidebar to navigate between pages:
    - **Landing** to get an overview
    - **Login** to sign in
    - **Recommendations** for personalized suggestions
    - **Game Info** to see details of a specific game
    - **Explanation** for a project explanation and analysis
    """
)

st.write("If you haven't already, choose a page from the sidebar on the left to begin!")
