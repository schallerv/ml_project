import argparse
from models import load_ratings_data, load_similarity_matrix, predict_ratings, get_recommendations_for_user
from train_eval import plot_predictions


# setup argparse to run from terminal
def main(plot_output):
    # Load ratings data for evaluation.
    ratings_df = load_ratings_data()

    # For demonstration, get recommendations for a test user.
    test_username = "testuser"  # Replace with an actual test username if available.
    recommended_games = get_recommendations_for_user(test_username, top_n=5, k=10)
    print("Recommendations for {}:".format(test_username))
    for game in recommended_games:
        print(game)

    # Also evaluate the prediction error on a held-out test set.
    test_ratings_df = ratings_df.sample(frac=0.2, random_state=42)
    sim_df = load_similarity_matrix()
    test_predictions_df = predict_ratings(test_ratings_df, ratings_df, sim_df, k=10)
    true_ratings = test_predictions_df["Rating"].values.tolist()
    predicted_ratings = test_predictions_df["Predicted"].values.tolist()
    plot_predictions(true_ratings, predicted_ratings, save_as=plot_output, smoothing_window=20)
    print(f"True vs Predicted plot saved to {plot_output}")
    test_predictions_df.to_csv("test_predictions_with_predicted.csv", index=False)
    print("Test predictions saved to 'test_predictions_with_predicted.csv'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Procedural Content-Based Recommendation Evaluation")
    parser.add_argument("--plot_output", type=str, default="true_vs_predicted.png",
                        help="File name for the true vs predicted plot output.")
    args = parser.parse_args()
    main(args.plot_output)
