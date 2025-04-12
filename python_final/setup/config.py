# cool toned color palette
# COLOR_SCHEME = {
#     'background': "#1F2C38",  # dark bluish gray
#     'primary': "#3E92CC",     # cool blue
#     'secondary': "#66A5AD",   # soft teal
#     'accent': "#A2B9BC",      # muted light blue/gray
#     'text': "#FFFFFF"         # white text for contrast
# }

BACKGROUND_COLOR = "#1F2C38"
PRIMARY_COLOR = "#3E92CC"
SECONDARY_COLOR = "#66A5AD"
TERNARY_COLOR = "#A2B9BC"
BLACK_COLOR = "#FFFFFF"

# data processing parameters
DATA_PARAMS = {
    'continuous_columns_end_index': 514  # first 514 columns are continuous; remaining columns are binary flags
}

# model hyperparameters
MODEL_PARAMS = {
    'latent_dim': 64,
    'encoder_layers': [1028, 512, 128],
    'decoder_layers': [128, 512, 1028],
    'noise_std': 0.2,
}

TRAINING_PARAMS = {
    'epochs': 50,
    'batch_size': 256,
    'learning_rate': .0001
}

GAMES_CSV_PATH = "../../bgg_data/overall_games_starter.csv"
GAME_EMBEDDINGS_CSV_PATH = "../../bgg_data/game_embeddings.csv"
GAMES_SIMILARITIES_CSV_PATH = "../../bgg_data/game_similarities.csv"
RATINGS_CSV_PATH = "../../bgg_data/ratings_starter.csv"
GAME_DESCRIPTIONS_CSV_PATH = "../../bgg_data/game_descriptions.csv"
GAME_COLUMNS_JSON_PATH = "../../bgg_data/columns.json"

AUTOENCODER_PATH = "denoising_autoencoder.pth"
