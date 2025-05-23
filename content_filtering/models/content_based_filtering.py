import pandas as pd
import numpy as np
from setup import RATINGS_CSV_PATH, GAMES_CSV_PATH, GAMES_SIMILARITIES_CSV_PATH


def load_ratings_data(ratings_csv=RATINGS_CSV_PATH):
    """
    loads rating data and normalizes usernames
    :param ratings_csv: path to ratings data
    :return: df of ratings data
    """
    df = pd.read_csv(ratings_csv)
    df["Username"] = df["Username"].str.strip().str.lower()
    return df


def load_games_data(games_csv=GAMES_CSV_PATH):
    """
    loads game data and standardizes bgg ids
    :param games_csv: path to games file
    :return: df of game data
    """
    df = pd.read_csv(games_csv)
    df['BGGId'] = df['BGGId'].astype(float).astype(int)
    return df


def load_similarity_matrix(sim_csv=GAMES_SIMILARITIES_CSV_PATH):
    """
    loads similarity scores file
    :param sim_csv: path to sim scores file
    :return: df of sim scores
    """
    sim_df = pd.read_csv(sim_csv, index_col=0)
    sim_df.index = sim_df.index.astype(float).astype(int)
    sim_df.columns = sim_df.columns.astype(float).astype(int)
    return sim_df


def predict_ratings(test_df, train_df, sim_df, k=10):
    """
    does content based filtering to predict all ratings missing in test df
    :param test_df: df with missing ratings to be populated
    :param train_df: df with data to predict missing ratings
    :param sim_df: df of similarity scores for all pairs of games
    :param k: max number of ratings to consider in weighted average
    :return: test df but with missing ratings populated
    """
    test_df = test_df.copy()
    test_df['BGGId'] = test_df['BGGId'].astype(float).astype(int)
    train_df = train_df.copy()
    train_df['BGGId'] = train_df['BGGId'].astype(float).astype(int)
    sim_df = sim_df.copy()
    sim_df.index = sim_df.index.astype(float).astype(int)
    sim_df.columns = sim_df.columns.astype(float).astype(int)
    game_to_idx = {game: i for i, game in enumerate(sim_df.index)}
    sim_matrix = sim_df.values
    # Pre-group training ratings by user.
    user_ratings_dict = {}
    for user, group in train_df.groupby('Username'):
        rated_ids = group['BGGId'].values
        ratings = group['Rating'].values.astype(float)
        valid_mask = np.array([gid in game_to_idx for gid in rated_ids])
        if np.any(valid_mask):
            user_ratings_dict[user] = (rated_ids[valid_mask], ratings[valid_mask])
        else:
            user_ratings_dict[user] = (np.array([]), np.array([]))
    predicted_ratings = []
    for _, row in test_df.iterrows():
        user = row['Username']
        target_game = int(row['BGGId'])
        if target_game not in game_to_idx:
            predicted_ratings.append(np.nan)
            continue
        target_idx = game_to_idx[target_game]
        if user not in user_ratings_dict or len(user_ratings_dict[user][0]) == 0:
            predicted_ratings.append(np.nan)
            continue
        rated_ids, ratings = user_ratings_dict[user]
        rated_indices = np.array([game_to_idx[gid] for gid in rated_ids])
        sims = sim_matrix[target_idx, rated_indices]
        if len(sims) > k:
            top_k_indices = np.argsort(sims)[-k:]
            sims = sims[top_k_indices]
            ratings = ratings[top_k_indices]
        if np.sum(sims) > 0:
            prediction = np.dot(sims, ratings) / np.sum(sims)
        else:
            prediction = np.nan
        predicted_ratings.append(prediction)
    test_df["Predicted"] = predicted_ratings
    return test_df


def get_recommendations_for_user(username, top_n=5, k=10):
    """
    get content based recs for a specific user
    :param username: user to get recs for
    :param top_n: number of recs to get
    :param k: max number of ratings to consider in weighted average
    :return: recs for given user
    """
    ratings_df = load_ratings_data()
    games_df = load_games_data()
    sim_df = load_similarity_matrix()
    username = username.strip().lower()

    # Check if the user has any ratings.
    user_ratings = ratings_df[ratings_df['Username'] == username]
    if user_ratings.empty:
        # For users with no ratings, return the most popular games based on NumUserRatings.
        popular_games = games_df.sort_values(by="NumUserRatings", ascending=False).head(top_n)
        return popular_games[['BGGId', 'Name']].to_dict(orient='records')

    # Exclude games already rated by the user.
    rated_games = set(user_ratings['BGGId'])
    candidates = games_df[~games_df['BGGId'].isin(rated_games)]
    if candidates.empty:
        return []

    candidate_df = candidates[['BGGId']].copy()
    candidate_df['Username'] = username
    candidate_df['Rating'] = np.nan  # Dummy value for compatibility.
    predicted_candidates = predict_ratings(candidate_df, ratings_df, sim_df, k=k)
    predicted_candidates = predicted_candidates.dropna(subset=["Predicted"])
    if predicted_candidates.empty:
        return []
    top_candidates = predicted_candidates.sort_values("Predicted", ascending=False).head(top_n)
    # Merge with games_df to retrieve game names.
    recommended = top_candidates.merge(games_df[['BGGId', 'Name']], on='BGGId', how='left')
    return recommended.to_dict(orient='records')

