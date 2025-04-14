# Collaborative and Content-Based Approaches to Board Game Recommendation Systems
# By: Victoria Schaller and Claudia Levi

This project builds a personalized board game recommendation system usingtwo different methods:  a **denoising autoencoder** with **content-based filtering** and **item-item collaborative filtering**. It is designed to help users discover new board games based on their prior ratings and a rich feature set describing each game.

## Collaborative Filtering
For collaborative filtering, data can be found at https://www.kaggle.com/datasets/threnjen/board-games-database-from-boardgamegeek
Download all csvs and combine them into a folder. Make sure to change data location references before running

## Content Based Filtering

### Setup

1. **Clone the repository and navigate into the project directory.**

2. **Ensure your environment is set up.**
   If you're using Conda, create an environment and install dependencies manually (see `requirements.txt` for pip users).

3. **Adjust File Paths.**
   Many scripts assume the BGG data is located in a folder named `bgg_data` located in the same directory as where the repo is located (not in the repo). You'll need to adjust the filepaths in the code to account for where you put bgg_data. See content_filtering/setup/config.py to adjust them for content filtering. 
   

### Running the App
In terminal, navigate to the content_filtering folder. Then run ```streamlit run app.py ```
You can login using an existing user, or you can make a new user. If you make a new user, you will need to dd some game ratings before the recommendations will become meaningful. For users who have not rated any games, initial recommendations are just the most popular games.
