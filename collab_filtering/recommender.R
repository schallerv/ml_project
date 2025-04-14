# DS4420 Final Project
# Victoria Schaller

################# imports ################# 
# libraries
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(Matrix) 
library(ggplot2)
library(purrr)
setwd("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project")

# data
subcategories <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/subcategories.csv", show_col_types = FALSE)
games <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/games.csv", show_col_types = FALSE)
ratings <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/user_ratings.csv", show_col_types = FALSE)

################# User interaction steps ################# 
# remove all games with average rating below 4
games <- games %>%
  filter(AvgRating >= quantile(AvgRating, 0.30))

# filter number of players
repeat {
  user_input <- readline("Enter the number of players: ")
  num_players <- as.numeric(user_input)
  
  # Check if the input is a valid single number
  if (!is.na(num_players) && length(num_players) == 1) {
    break  # Exit the loop if the input is valid
  } else {
    cat("Invalid input. Please enter exactly one number.\n")
  }
}

cat("You entered:", num_players, "\n")

games <- filter(games, MaxPlayers >= num_players)
games <- filter(games, MinPlayers <= num_players)

# rename category columns
colnames(games)[41:48] <- gsub("Cat:", "", colnames(games)[41:48])

# filter columns to id, name, and category data
games <- games %>% select(1, 2, 6, 41:48)

# join games and subcategory data
full_games <- inner_join(games, subcategories, by = "BGGId")

# categories list and count categories per game
full_games <- full_games %>% 
  mutate(Categories = apply(select(., 4:11), 1, function(row) paste(names(.)[4:11][row == 1], collapse = ", ")),
         Num_Categories = rowSums(select(., 4:11)))

# subcategories list and count subcategories per game
full_games <- full_games %>% 
  mutate(Sub_Categories = apply(select(., 12:21), 1, function(row) paste(names(.)[12:21][row == 1], collapse = ", ")),
         Num_Sub_Categories = rowSums(select(., 12:21)))

# filter games that are not in any category
full_games <- filter(full_games, Num_Categories > 0)

# extract unique categories and print
unique_categories <- sort(unique(unlist(strsplit(na.omit(full_games$Categories), ", "))))
cat("Categories:\n")
cat(paste0(seq_along(unique_categories), ". ", unique_categories), sep = "\n")

# allow user to select categories of interest
repeat {
  user_input <- readline("Select categories of interest by number(s) separated by a space: ")
  selected_categories <- as.integer(strsplit(user_input, " ")[[1]])
  
  # Remove nonnumeric values and check if there's at least one valid entry
  selected_categories <- selected_categories[!is.na(selected_categories)]
  valid_categories <- selected_categories[selected_categories %in% seq_along(unique_categories)]
  
  # Exit the loop if at least one number is entered
  if (length(valid_categories) > 0) {
    break  
  } else {
    cat("Invalid input. Please enter at least one number.\n")
  }
}
category_names <- unique_categories[valid_categories]
print("You Have Selected:")
print(category_names)
filtered_games <- full_games %>% filter(if_any(all_of(category_names), ~ . == 1))

# allow user to select subcategories of interest

unique_sub_categories <- sort(unique(unlist(strsplit(na.omit(filtered_games$Sub_Categories), ", "))))
cat("Subcategories:\n")
cat(paste0(seq_along(unique_sub_categories), ". ", unique_sub_categories), sep = "\n")
repeat {
  user_input <- readline("Select subcategories of interest by number(s) separated by a space: ")
  selected_subcategories <- as.integer(strsplit(user_input, " ")[[1]])
    
  # Remove nonnumeric values and check if there's at least one valid entry
  selected_subcategories <- selected_subcategories[!is.na(selected_subcategories)]
  print(selected_subcategories)
  valid_subcategories <- selected_subcategories[selected_subcategories %in% seq_along(unique_sub_categories)]
  print(valid_subcategories)
  
  # Exit the loop if at least one number is entered
  if (length(valid_subcategories) > 0) {
    break
  } else {
    cat("Invalid input. Please enter at least one number.\n")
  }
}
sub_category_names <- unique_sub_categories[valid_subcategories]
print("You Have Selected:")
print(sub_category_names)
filtered_games <- filtered_games %>% filter(if_any(all_of(sub_category_names), ~ . == 1))

# reset index
filtered_games <- filtered_games %>% arrange(desc(AvgRating))
filtered_games <- filtered_games %>% mutate(Game_Number = row_number())

# display games in a new window
View(filtered_games %>% select(Name))


# user gives numbers of favorite games
fav_numbers <- as.integer(strsplit(readline("Enter favorite game numbers separated by a space: "), " ")[[1]])
fav_games <- filtered_games %>% filter(Game_Number %in% fav_numbers)
#print(fav_games)

# user gives numbers of least favorite games
least_fav_numbers <- as.integer(strsplit(readline("Enter least favorite game numbers separated by a space: "), " ")[[1]])
least_fav_games <- filtered_games %>% filter(Game_Number %in% least_fav_numbers)
#print(least_fav_games)

# append new ratings from input to ratings data
new_ratings <- data.frame(
  BGGId = c(fav_games$BGGId, least_fav_games$BGGId),
  Rating = c(rep(10, nrow(fav_games)), rep(1, nrow(least_fav_games))),
  Username = rep("TestUser", nrow(fav_games) + nrow(least_fav_games))
)

ratings <- bind_rows(ratings, new_ratings)

# filter rating data 
ratings <- ratings %>% filter(BGGId %in% filtered_games$BGGId)

################# filtering ################# 
# users and items
user_levels <- unique(ratings$Username)
item_levels <- unique(ratings$BGGId)

# get numeric index
ratings$UserIndex <- match(ratings$Username, user_levels)
ratings$ItemIndex <- match(ratings$BGGId, item_levels)

# sparse matrix w/ rows -> users, columns -> items
Rmat <- sparseMatrix(
  i = ratings$UserIndex,
  j = ratings$ItemIndex,
  x = ratings$Rating,
  dims = c(length(user_levels), length(item_levels))
)
rownames(Rmat) <- user_levels
colnames(Rmat) <- item_levels

# cosine similarity
cosine_sim_sp <- function(vec1, vec2) {
  # find entries 
  ind1 <- which(vec1 != 0)
  ind2 <- which(vec2 != 0)
  common <- intersect(ind1, ind2)
  if (length(common) == 0) return(0)
  # get cosine sim
  dot_prod <- sum(vec1[common] * vec2[common])
  norm1 <- sqrt(sum(vec1[common]^2))
  norm2 <- sqrt(sum(vec2[common]^2))
  if (norm1 == 0 || norm2 == 0) return(0)
  return(dot_prod / (norm1 * norm2))
}

# Function to make predictions for a given user
make_predictions <- function(user_id, Rmat, candidate_items = NULL) {
  # Get user's ratings
  user_ratings <- Rmat[user_id, ]
  rated_items <- colnames(Rmat)[user_ratings != 0]
  
  # If no candidate items provided, use all unrated items
  if (is.null(candidate_items)) {
    candidate_items <- colnames(Rmat)[user_ratings == 0]
  }
  
  # Make predictions for each candidate item
  predictions <- sapply(candidate_items, function(item) {
    # For each game user rated, get similarity with candidate item
    sims <- sapply(rated_items, function(r_item) {
      cosine_sim_sp(Rmat[, item], Rmat[, r_item])
    })
    # If all similarities are zero, return NA
    if (sum(abs(sims)) == 0) return(NA)
    # Get final prediction
    pred <- sum(sims * user_ratings[rated_items]) / sum(abs(sims))
    return(pred)
  })
  
  # Remove items with NA ratings
  predictions <- predictions[!is.na(predictions)]
  # Sort ratings
  predictions <- sort(predictions, decreasing = TRUE)
  
  return(predictions)
}

# TestUser's ratings
test_ratings <- Rmat["TestUser", ]
rated_items <- colnames(Rmat)[test_ratings != 0]
candidate_items <- colnames(Rmat)[test_ratings == 0]

# Get predictions for TestUser
predictions <- make_predictions("TestUser", Rmat)

# top 5 recommendations (or all if fewer than 5)
top_n <- if (length(predictions) >= 5) head(predictions, 5) else predictions

################# recommendations ################# 
# result dataframe
result <- data.frame(BGGId = names(top_n), PredictedRating = top_n, row.names = NULL)

# make same data type for join
result <- result %>% mutate(BGGId = as.character(BGGId))
games <- games %>% mutate(BGGId = as.character(BGGId))

# Join result with games to get the Name column
result_with_names <- result %>%
  left_join(games, by = "BGGId")

print("Your Recommended Games:")
for (i in 1:nrow(result_with_names)) {
  print(result_with_names$Name[i])
}

################# model evaluation ################# 
cat("\nEvaluating Model Accuracy\n")

# Filter users with at least 10 ratings, need to have good amount of data to measure accuracy
user_counts <- ratings %>% 
  group_by(Username) %>% 
  tally() %>% 
  filter(n >= 10)

# split ratings into training and test
split_user_ratings <- function(user_ratings, train_prop = 0.8) {
  n_ratings <- nrow(user_ratings)
  train_size <- floor(n_ratings * train_prop)
  train_indices <- sample(1:n_ratings, train_size)
  
  list(
    train = user_ratings[train_indices, , drop = FALSE],
    test = user_ratings[-train_indices, , drop = FALSE]
  )
}

# filter users to users who passed the filter
ratings_filtered <- ratings %>%
  filter(Username %in% user_counts$Username) %>%
  select(Username, BGGId, Rating)

# Split each user's ratings into training and test sets
set.seed(42)
split_results <- ratings_filtered %>%
  group_by(Username) %>%
  group_split() %>%
  map(~ split_user_ratings(.x))

# make mast train and test sets
ratings_train <- map_dfr(split_results, ~ .x$train)
ratings_test <- map_dfr(split_results, ~ .x$test)

# make user and item index mappings for matrix construction
ratings_train <- ratings_train %>%
  transmute(
    Username = as.character(Username),
    BGGId = as.character(BGGId),
    Rating = as.numeric(Rating)
  ) %>%
  filter(!is.na(Username), !is.na(BGGId), !is.na(Rating))

ratings_test <- ratings_test %>%
  transmute(
    Username = as.character(Username),
    BGGId = as.character(BGGId),
    Rating = as.numeric(Rating)
  ) %>%
  filter(!is.na(Username), !is.na(BGGId), !is.na(Rating))

# get indexs for matrix
train_user_levels <- unique(ratings_train$Username)
train_item_levels <- unique(ratings_train$BGGId)

# add index for users and items
ratings_train <- ratings_train %>%
  mutate(
    UserIndex = match(Username, train_user_levels),
    ItemIndex = match(BGGId, train_item_levels)
  ) %>%
  filter(!is.na(UserIndex), !is.na(ItemIndex))

# build matrix
Rmat_train <- sparseMatrix(
  i = ratings_train$UserIndex,
  j = ratings_train$ItemIndex,
  x = ratings_train$Rating,
  dims = c(length(train_user_levels), length(train_item_levels))
)
rownames(Rmat_train) <- train_user_levels
colnames(Rmat_train) <- train_item_levels

# predict ratings for a given user
predict_user_ratings <- function(user_id) {
  # skip if user not in training data
  if (!(user_id %in% rownames(Rmat_train))) {
    cat("User", user_id, "not found in training data\n")
    return(NULL)
  }
  
  # get test items for this user in training data
  test_items <- ratings_test %>% 
    filter(Username == user_id) %>% 
    filter(BGGId %in% colnames(Rmat_train)) %>%  # Only keep items that exist in training
    pull(BGGId)
  
  if (length(test_items) == 0) {
    cat("No valid test items for user", user_id, "\n")
    return(NULL)
  }
  
  # Get predictions using the make_predictions function
  predictions <- make_predictions(user_id, Rmat_train, test_items)
  
  # actual ratings for the items w predictions
  actual_ratings <- ratings_test %>% 
    filter(Username == user_id, BGGId %in% test_items) %>% 
    pull(Rating)
  
  # return actual vs. predicted
  tibble(
    UserId = user_id,
    BGGId = test_items,
    ActualRating = actual_ratings,
    PredictedRating = predictions
  )
}

# Evaluate predictions for a sample of users
sample_users <- sample(user_counts$Username, min(100, nrow(user_counts)))
cat("Evaluating predictions for", length(sample_users), "users\n")

all_predictions <- map_dfr(sample_users, predict_user_ratings)

# Remove any rows with NA predictions
all_predictions <- all_predictions %>% filter(!is.na(PredictedRating))

if (nrow(all_predictions) > 0) {
  # Calculate RMSE
  rmse <- sqrt(mean((all_predictions$PredictedRating - all_predictions$ActualRating)^2, na.rm = TRUE))
  cat("\nRoot Mean Square Error (RMSE):", round(rmse, 3), "\n")
  
  # subtitle
  plot_subtitle <- sprintf("(Players: %d, Categories: %s, Subcategories: %s)", 
                          num_players,
                          paste(category_names, collapse = ", "),
                          if(exists("sub_category_names")) paste(sub_category_names, collapse = ", ") else "None")
  
  # file name
  file_name <- sprintf("rating_predictions_%dplayers_%s%s.png",
                      num_players,
                      paste(category_names, collapse = "_"),
                      if(exists("sub_category_names")) paste("_", paste(sub_category_names, collapse = "_"), sep = "") else "")
  
  #plot real vs predicted
  p <- ggplot(all_predictions, aes(x = ActualRating, y = PredictedRating)) +
    geom_point(alpha = 0.4) +
    geom_abline(slope = 1, intercept = 0, color = "red", linetype = "dashed") +
    labs(title = "True Ratings vs. Predicted Ratings",
         subtitle = plot_subtitle,
         x = "Actual Rating", 
         y = "Predicted Rating") +
    theme_minimal() +
    theme(
      plot.background = element_rect(fill = "white"),
      panel.background = element_rect(fill = "white"),
      plot.title = element_text(face = "bold", hjust = 0.5),
      plot.subtitle = element_text(hjust = 0.5)
    )
  
  # Save the plot
  ggsave(file_name, p, width = 8, height = 6, bg = "white")
  cat("\nPlot has been saved as '", file_name, "' in your working directory.\n", sep = "")
  
  # Display the plot
  print(p)
}
