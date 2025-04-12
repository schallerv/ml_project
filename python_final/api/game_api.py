import json
import pandas as pd
from setup import RATINGS_CSV_PATH, GAMES_CSV_PATH, GAMES_SIMILARITIES_CSV_PATH, GAME_DESCRIPTIONS_CSV_PATH, GAME_COLUMNS_JSON_PATH
from models import get_recommendations_for_user

class GameAPI:
    def __init__(self, ratings_csv=RATINGS_CSV_PATH, games_csv=GAMES_CSV_PATH,
                 games_similarity_csv=GAMES_SIMILARITIES_CSV_PATH):
        # Load ratings and store normalized usernames.
        self.ratings_df = pd.read_csv(ratings_csv)
        self.ratings_df["Username"] = self.ratings_df["Username"].str.strip().str.lower()
        self.users = sorted(self.ratings_df["Username"].unique().tolist())
        # Load games data.
        self.games_df = pd.read_csv(games_csv)
        self.games_df['BGGId'] = self.games_df['BGGId'].astype(float).astype(int)
        # Load similarity matrix.
        self.similarity_df = pd.read_csv(games_similarity_csv, index_col=0)
        self.similarity_df.index = self.similarity_df.index.astype(float).astype(int)
        self.similarity_df.columns = self.similarity_df.columns.astype(float).astype(int)
        # For easy lookup.
        self.game_ids_names = self.games_df[["BGGId", "Name"]]

        # Get game descriptions.
        self.game_descriptions_df = pd.read_csv(GAME_DESCRIPTIONS_CSV_PATH)
        # Keep only rows with a BGGId present in the games data.
        self.game_descriptions_df = self.game_descriptions_df[self.game_descriptions_df['BGGId'].isin(self.game_ids_names['BGGId'])]

        # Get game columns configuration.
        with open(GAME_COLUMNS_JSON_PATH, 'r') as f:
            self.game_columns = json.load(f)

    # Existing methods (get_users, add_user, get_ratings, etc.) remain here.

    def get_game_info(self, bggid):
        """
        Retrieves a dictionary of game information for the given BGGId.
        Merges basic game info with the game description if available.
        """
        bggid = int(bggid)
        game_row = self.games_df[self.games_df["BGGId"] == bggid]
        if game_row.empty:
            return None
        # Convert the DataFrame row to a dictionary.
        game_info = game_row.to_dict(orient="records")[0]
        # Add the description if available.
        description_row = self.game_descriptions_df[self.game_descriptions_df["BGGId"] == bggid]
        if not description_row.empty:
            description_info = description_row.to_dict(orient="records")[0]
            game_info["Description"] = description_info.get("Description", "")
        else:
            game_info["Description"] = ""
        return game_info

    def get_users(self):
        return self.users

    def add_user(self, username):
        username = username.strip().lower()
        if username not in self.users:
            self.users.append(username)
        return username

    def get_ratings(self):
        return self.ratings_df

    def get_user_ratings(self, username):
        username = username.strip().lower()
        return self.ratings_df[self.ratings_df["Username"] == username]

    def get_games(self):
        return self.games_df

    def get_recommendations(self, username, top_n=5, k=10):
        return get_recommendations_for_user(username, top_n=top_n, k=k)

    def add_rating(self, username, bggid, rating):
        """
        Adds a new rating for the given username and game.
        Returns a tuple (success: bool, message: str).
        """
        username = username.strip().lower()
        bggid = int(bggid)
        # Check if the rating already exists.
        if not self.ratings_df[(self.ratings_df["Username"] == username) & (self.ratings_df["BGGId"] == bggid)].empty:
            return False, "You have already rated this game."
        # Check if the game exists.
        if self.games_df[self.games_df["BGGId"] == bggid].empty:
            return False, "Game does not exist."
        # Prepare the new rating entry.
        new_rating = {"BGGId": bggid, "Rating": rating, "Username": username}
        new_rating_df = pd.DataFrame([new_rating])
        self.ratings_df = pd.concat([self.ratings_df, new_rating_df], ignore_index=True)
        # Save the updated ratings to file.
        try:
            self.ratings_df.to_csv(RATINGS_CSV_PATH, index=False)
            return True, "Rating added successfully."
        except Exception as e:
            return False, f"Error saving rating: {e}"

    def get_high_rated_merged_data(self, username, threshold):
        """
        Returns a merged DataFrame of the user's ratings (for games rated >= threshold)
        combined with game information from the games dataframe.
        """
        username = username.strip().lower()
        user_ratings = self.get_user_ratings(username)
        high_rated = user_ratings[user_ratings["Rating"] >= threshold]
        merged = high_rated.merge(self.games_df, on="BGGId", how="left")
        return merged

