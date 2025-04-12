import streamlit as st
import matplotlib.pyplot as plt

def filter_games_by_tags(games_df, selected_tags):
    """
    Filter games based on a list of selected binary tag columns.
    For example, if 'Deck Building' is a column, only return rows where that flag == 1.
    """
    if not selected_tags:
        return games_df
    for tag in selected_tags:
        games_df = games_df[games_df[tag] == 1]
    return games_df


def plot_true_vs_predicted(true_values, predicted_values):
    fig, ax = plt.subplots()
    ax.scatter(true_values, predicted_values, alpha=0.5, c="blue")
    ax.plot([min(true_values), max(true_values)], [min(true_values), max(true_values)], c="red", linestyle="--")
    ax.set_xlabel("True Values")
    ax.set_ylabel("Predicted Values")
    ax.set_title("True vs. Predicted Plot")
    return fig
