# DS4420 Final Project
# Victoria Schaller

################# imports ################# 
# libraries
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(Matrix) 

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
  
  # Remove NA values (which occur if the input isn't a number) and check if there's at least one valid number
  selected_categories <- selected_categories[!is.na(selected_categories)]
  valid_categories <- selected_categories[selected_categories %in% seq_along(unique_categories)]
  
  if (length(valid_categories) > 0) {
    break  # Exit the loop if at least one number is entered
  } else {
    cat("Invalid input. Please enter at least one number.\n")
  }
}
category_names <- unique_categories[valid_categories]
print("You Have Selected:")
print(category_names)
filtered_games <- full_games %>% filter(if_any(all_of(category_names), ~ . == 1))

# allow user to select subcategories of interest
y_n <- readline("Would you like to select a subcategory? (Y/N) ")
if (tolower(y_n) == "y") {
  unique_sub_categories <- sort(unique(unlist(strsplit(na.omit(filtered_games$Sub_Categories), ", "))))
  cat("Subcategories:\n")
  cat(paste0(seq_along(unique_sub_categories), ". ", unique_sub_categories), sep = "\n")
  repeat {
    user_input <- readline("Select subcategories of interest by number(s) separated by a space: ")
    selected_subcategories <- as.integer(strsplit(user_input, " ")[[1]])
    
    # Remove NA values (which occur if the input isn't a number) and check if there's at least one valid number
    selected_subcategories <- selected_subcategories[!is.na(selected_subcategories)]
    print(selected_subcategories)
    valid_subcategories <- selected_subcategories[selected_subcategories %in% seq_along(unique_sub_categories)]
    print(valid_subcategories)
    
    if (length(valid_subcategories) > 0) {
      break  # Exit the loop if at least one number is entered
    } else {
      cat("Invalid input. Please enter at least one number.\n")
    }
  }
  sub_category_names <- unique_sub_categories[valid_subcategories]
  print("You Have Selected:")
  print(sub_category_names)
  filtered_games <- filtered_games %>% filter(if_any(all_of(sub_category_names), ~ . == 1))
}

print("SUBCAT CHOICE FILTER")
print(nrow(filtered_games))

# reset index
filtered_games <- filtered_games %>% arrange(desc(AvgRating))
filtered_games <- filtered_games %>% mutate(Game_Number = row_number())

# display games in a new window
View(filtered_games %>% select(Name, Game_Number))
#View(filtered_games %>% arrange(desc(AvgRating)) %>% select(Name, Game_Number))


# user gives numbers of favorite games
fav_numbers <- as.integer(strsplit(readline("Enter favorite game numbers separated by a space: "), " ")[[1]])
fav_games <- filtered_games %>% filter(Game_Number %in% fav_numbers)
print(fav_numbers)
print(fav_games)
# user gives numbers of least favorite games
least_fav_numbers <- as.integer(strsplit(readline("Enter least favorite game numbers separated by a space: "), " ")[[1]])
least_fav_games <- filtered_games %>% filter(Game_Number %in% least_fav_numbers)
print(least_fav_numbers)
print(least_fav_games)
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
# factor levels for users and items
user_levels <- unique(ratings$Username)
item_levels <- unique(ratings$BGGId)

# Map to numeric index
ratings$UserIndex <- match(ratings$Username, user_levels)
ratings$ItemIndex <- match(ratings$BGGId, item_levels)

# build sparse matrix: rows = users, columns = items
Rmat <- sparseMatrix(
  i = ratings$UserIndex,
  j = ratings$ItemIndex,
  x = ratings$Rating,
  dims = c(length(user_levels), length(item_levels))
)
rownames(Rmat) <- user_levels
colnames(Rmat) <- item_levels

# get TestUser's ratings
test_ratings <- Rmat["TestUser", ]
rated_items <- colnames(Rmat)[test_ratings != 0]
print(test_ratings)
candidate_items <- colnames(Rmat)[test_ratings == 0]

# cosine similarity function for sparse vectors
cosine_sim_sp <- function(vec1, vec2) {
  # indices of nonzero entries 
  ind1 <- which(vec1 != 0)
  ind2 <- which(vec2 != 0)
  common <- intersect(ind1, ind2)
  if (length(common) == 0) return(0)
  # Compute dot product and norms
  dot_prod <- sum(vec1[common] * vec2[common])
  norm1 <- sqrt(sum(vec1[common]^2))
  norm2 <- sqrt(sum(vec2[common]^2))
  if (norm1 == 0 || norm2 == 0) return(0)
  return(dot_prod / (norm1 * norm2))
}

# predicted ratings for each game
predictions <- sapply(candidate_items, function(item) {
  # For each game TestUser rated, cosine similarity with candidate item
  sims <- sapply(rated_items, function(r_item) {
    cosine_sim_sp(Rmat[, item], Rmat[, r_item])
  })
  # if all similarities are zero, null
  if (sum(abs(sims)) == 0) return(NA)
  # prediction using TestUser's ratings
  pred <- sum(sims * test_ratings[rated_items]) / sum(abs(sims))
  return(pred)
})

# remove items with NA ratings
predictions <- predictions[!is.na(predictions)]
# sort ratings in descending order
predictions <- sort(predictions, decreasing = TRUE)
# top 5 recommendations (or all if fewer than 5)
top_n <- if (length(predictions) >= 5) head(predictions, 5) else predictions

################# recommendations ################# 
# result dataframe
result <- data.frame(BGGId = names(top_n), PredictedRating = top_n, row.names = NULL)
print(result)

# mutate
result <- result %>% mutate(BGGId = as.character(BGGId))
games <- games %>% mutate(BGGId = as.character(BGGId))

# Join result with games to get the Name column
result_with_names <- result %>%
  left_join(games, by = "BGGId")

print("Your Recommended Games:")
# Print each row with its associated Name
for (i in 1:nrow(result_with_names)) {
  print(result_with_names$Name[i])
}
