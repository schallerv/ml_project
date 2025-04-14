import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import altair as alt
import math
from setup import BACKGROUND_COLOR, PRIMARY_COLOR, SECONDARY_COLOR, BLACK_COLOR

from api import GameAPI


def dashboard_app():
    """
    streamlit implementation of the user dashboard page
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

    # Ensure the user is logged in.
    if "username" not in st.session_state or st.session_state["username"] == "":
        st.error("Please log in to access the dashboard.")
        st.stop()
    username = st.session_state["username"]

    #st.set_page_config(page_title="User Dashboard")
    st.title("User Dashboard")
    st.markdown(f"Welcome, **{username}**!")

    # Get or create the API instance.
    if "api" not in st.session_state:
        st.session_state["api"] = GameAPI()
    api = st.session_state["api"]

    # --- Rating Distribution ---
    st.subheader("Your Rating Distribution")
    user_ratings = api.get_user_ratings(username)
    if user_ratings.empty:
        st.info("You have not rated any games yet.")
    else:
        fig, ax = plt.subplots(facecolor=BACKGROUND_COLOR, figsize=(8, 4))
        ax.hist(user_ratings["Rating"], bins=10, range=(0, 10), color=PRIMARY_COLOR, edgecolor=BLACK_COLOR)
        ax.set_xlabel("Rating", color=BLACK_COLOR)
        ax.set_ylabel("Count", color=BLACK_COLOR)
        ax.set_title("Rating Distribution", color=BLACK_COLOR)
        ax.tick_params(colors=BLACK_COLOR)
        st.pyplot(fig)

    # --- Set Threshold for Visualizations ---
    st.subheader("Visualizations")
    st.markdown("Use the slider below to set the minimum rating to consider for all visualizations.")
    threshold = st.slider("Rating Threshold", min_value=0.0, max_value=10.0, value=7.0, step=0.5)

    # Get filtered, merged data via the API method.
    merged_high = api.get_high_rated_merged_data(username, threshold)

    if merged_high.empty:
        st.info("No games rated above the selected threshold.")
    else:
        # --- Common Game Themes Visualization ---
        theme_cols = [col for col in merged_high.columns if col.startswith("Cat:")]
        if theme_cols:
            theme_counts = merged_high[theme_cols].sum().sort_values(ascending=False)
            st.markdown("### Common Game Categories")
            # Slider to control number of bars (default 8).
            num_theme_bars = st.slider("Select number of bars to display for Game Categories", min_value=1, max_value=20, value=8,
                                       key="theme_bars")
            theme_chart_data = theme_counts.reset_index()
            theme_chart_data.columns = ["Theme", "Count"]
            theme_chart_data = theme_chart_data.head(num_theme_bars)
            theme_chart = alt.Chart(theme_chart_data).mark_bar(color=PRIMARY_COLOR).encode(
                x=alt.X("Count:Q"),
                y=alt.Y("Theme:N", sort='-x')
            ).properties(width=600, background=BACKGROUND_COLOR).configure_axis(labelColor=BLACK_COLOR, titleColor=BLACK_COLOR)
            st.altair_chart(theme_chart)
        else:
            st.info("No theme data available in your games.")

        # --- Game Feature Visualizations for Each Feature Type ---
        for feature_type, flag_list in api.game_columns.items():
            st.markdown(f"#### {feature_type}")
            flag_counts = {}
            for flag in flag_list:
                if flag in merged_high.columns:
                    try:
                        flag_counts[flag] = merged_high[flag].astype(int).sum()
                    except Exception:
                        flag_counts[flag] = 0
                else:
                    flag_counts[flag] = 0
            if flag_counts:
                chart_df = pd.DataFrame(list(flag_counts.items()), columns=["Flag", "Count"])
                chart_df = chart_df.sort_values(by="Count", ascending=False)
                # Slider per feature category for number of bars.
                num_bars = st.slider(f"Select number of bars to display for {feature_type}", min_value=1, max_value=20,
                                     value=8, key=f"bars_{feature_type}")
                chart_df = chart_df.head(num_bars)
                chart = alt.Chart(chart_df).mark_bar(color=PRIMARY_COLOR).encode(
                    x=alt.X("Count:Q"),
                    y=alt.Y("Flag:N", sort='-x')
                ).properties(width=600, background=BACKGROUND_COLOR).configure_axis(labelColor=BLACK_COLOR, titleColor=BLACK_COLOR)
                st.altair_chart(chart)
            else:
                st.info(f"No data available for feature type: {feature_type}")

    # --- Add New Rating (Now with Searchable Selectbox) ---
    st.subheader("Add a New Rating")
    with st.form(key="new_rating_form"):
        # Use a selectbox for the user to choose a game.
        games_df = api.get_games()[["BGGId", "Name"]].drop_duplicates(subset=["Name"]).sort_values("Name")
        # Convert the DataFrame rows to namedtuple objects so that we can leverage a format function.
        games_options = list(games_df.itertuples(index=False, name="Game"))
        selected_game = st.selectbox("Select a Game", options=games_options,
                                     format_func=lambda x: f"{x.Name} (BGGId: {x.BGGId})")
        new_rating = st.number_input("Rating (0-10):", min_value=0.0, max_value=10.0, value=5.0, step=0.5)
        submit = st.form_submit_button("Submit Rating")
    if submit:
        # Check if user already rated this game.
        if not user_ratings[user_ratings["BGGId"] == selected_game.BGGId].empty:
            st.error("You have already rated this game.")
        else:
            success, message = api.add_rating(username, selected_game.BGGId, new_rating)
            if success:
                st.success(message)
                # Refresh the ratings after adding.
                user_ratings = api.get_user_ratings(username)
            else:
                st.error(message)

    # --- Paginated Table of Rated Games (At the Bottom) ---
    st.subheader("Your Rated Games")
    if user_ratings.empty:
        st.info("No ratings to display.")
    else:
        games_df = api.get_games()[["BGGId", "Name"]]
        merged_df = pd.merge(user_ratings, games_df, on="BGGId", how="left")
        merged_df = merged_df.sort_values(by="Rating", ascending=False)

        # Pagination parameters.
        page_size = 10
        total_rows = merged_df.shape[0]
        total_pages = math.ceil(total_rows / page_size)

        if "dashboard_page" not in st.session_state:
            st.session_state.dashboard_page = 1
        current_page = st.session_state.dashboard_page

        start_idx = (current_page - 1) * page_size
        end_idx = start_idx + page_size
        paged_df = merged_df.iloc[start_idx:end_idx]

        # Table header.
        header_cols = st.columns([0.6, 0.2, 0.2])
        header_cols[0].markdown("**Game**")
        header_cols[1].markdown("**Rating**")
        header_cols[2].markdown("**View Info**")

        # Table rows.
        for idx, row in paged_df.iterrows():
            row_cols = st.columns([0.6, 0.2, 0.2])
            row_cols[0].write(row["Name"] if pd.notnull(row["Name"]) else "Unknown Game")
            row_cols[1].write(row["Rating"])
            if row_cols[2].button("View Info", key=f"view_info_{row['BGGId']}_{idx}"):
                st.session_state["selected_game_id"] = row["BGGId"]
                st.switch_page("pages/game_page.py")

        col_prev, col_page, col_next = st.columns(3)
        if col_prev.button("Previous") and current_page > 1:
            st.session_state.dashboard_page = current_page - 1
            st.rerun()
        if col_next.button("Next") and current_page < total_pages:
            st.session_state.dashboard_page = current_page + 1
            st.rerun()
        col_page.write(f"Page {current_page} of {total_pages}")


if __name__ == "__main__":
    dashboard_app()
