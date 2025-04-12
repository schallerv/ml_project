import streamlit as st
from setup import BACKGROUND_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, TERNARY_COLOR, BLACK_COLOR
from api import GameAPI

# st.set_page_config(page_title="Login")

def login_app():
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

    # Load the API object once and store it in session state.
    if "api" not in st.session_state:
        st.session_state["api"] = GameAPI()

    # Use the API to retrieve the list of users.
    api = st.session_state["api"]
    existing_users = api.get_users()

    if "username" not in st.session_state:
        st.session_state["username"] = ""
    if "confirm_new_account" not in st.session_state:
        st.session_state["confirm_new_account"] = False

    st.title("Welcome to the Board Game Recommender")
    st.markdown(
        """
        **Instructions:**

        - Enter your username to log in.
        - If the username exists, you will be logged in.
        - If not, you can create a new account.
        """
    )

    input_username = st.text_input("Enter your username:")

    if st.button("Login"):
        if input_username.strip() == "":
            st.error("Please enter a valid username.")
        else:
            input_username = input_username.strip().lower()
            if input_username in existing_users:
                st.session_state["username"] = input_username
                st.success(f"Welcome back, {input_username}!")
                # Navigate to the landing page
                st.switch_page("pages/landing_page.py")
            else:
                # The user doesn't exist, so we let them confirm account creation
                st.session_state["confirm_new_account"] = True

    if st.session_state.get("confirm_new_account", False):
        st.warning("The username does not exist. Would you like to create a new account?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes, create new account"):
                new_user = api.add_user(input_username)
                st.session_state["username"] = new_user
                st.success(f"New account created. Welcome, {new_user}!")
                st.session_state["confirm_new_account"] = False
                st.switch_page("pages/landing_page.py")  # Go to landing page
        with col2:
            if st.button("No, try again"):
                st.info("Please enter a different username.")
                st.session_state["confirm_new_account"] = False

if __name__ == "__main__":
    login_app()
