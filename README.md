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
   1. If you're using Conda, run ```conda create --name myenv python=3.8.8```. Be sure to use python 3.8.8 to avoid dependancy issues.
   2. In terminal, navigate to content_filtering
   3. Run ```pip install -r requirements.txt ```

3. **Adjust File Paths.**
   If you move the bgg_data folder from the content_filtering folder or use the zipped version, you'll need to adjust the filepaths in the code to account for where you put bgg_data. See content_filtering/setup/config.py to adjust them for content filtering. 
   

### Running the App
1. In terminal, navigate to the content_filtering folder.
2. Run ```streamlit run app.py ```
3. You can login using an existing user, or you can make a new user. If you make a new user, you will need to add some game ratings before the recommendations become meaningful. For users who have not rated any games, initial recommendations are just the most popular games.
  - For a test user to see quality recommendations in action, we recommend "leffe dubbel". This user likes war strategy games, and their recommendations reflect this. Additionally, they have rated many games and thus have a rich user dashboard. 
