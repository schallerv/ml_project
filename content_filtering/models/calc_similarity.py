import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity


def standardize_row(row):
    """
    name is self-explanatory. This standardizes a row of data to range from 0 to 1
    :param row: row to be standardized
    :return: standardized row
    """
    row_min = row.min()
    row_max = row.max()
    if row_max - row_min == 0:
        return row  # Avoid division by zero if row is constant.
    return (row - row_min) / (row_max - row_min)


def compute_similarity_matrix(latent_csv_path, output_csv_path=None):
    """
    computes cosin similarity among all pairs of games in the latent csv file
    :param latent_csv_path: path to embeddings data
    :param output_csv_path: path to save sim scores at
    :return: None
    """
    df = pd.read_csv(latent_csv_path)
    bgg_ids = df['BGGId']
    features = df.drop(columns=['BGGId', 'Name'], errors='ignore').values

    similarity = cosine_similarity(features)
    sim_df = pd.DataFrame(similarity, index=bgg_ids, columns=bgg_ids)

    sim_df = sim_df.apply(standardize_row, axis=1)

    if output_csv_path:
        sim_df.to_csv(output_csv_path)
        print(f"Saved similarity matrix to {output_csv_path}")

    return sim_df


