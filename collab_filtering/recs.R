# DS4420 Final Project
# Victoria Schaller

# import libraries
library(dplyr)
library(tidyr)
library(readr)
library(tibble)
library(Matrix) 

# read in data
subcategories <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/subcategories.csv", show_col_types = FALSE)
games <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/games.csv", show_col_types = FALSE)
ratings <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/user_ratings.csv", show_col_types = FALSE)

################# User interaction steps ################# 
# remove all games with average rating below 4
games <- games %>%
  filter(AvgRating >= quantile(AvgRating, 0.30))

print("RATING FILTER")
print(nrow(games))

# filter number of players
num_players <- as.numeric(readline("Enter the number of players: "))  
cat("Only showing games for ", num_players, "players\n")  
games <- filter(games, MaxPlayers >= num_players)
games <- filter(games, MinPlayers <= num_players)

print("PLAYERS FILTER")
print(nrow(games))

# rename category columns
colnames(games)[41:48] <- gsub("Cat:", "", colnames(games)[41:48])

# filter columns to id, name, and category data
games <- games %>% select(1, 2, 41:48)

# join games and subcategory data
full_games <- inner_join(games, subcategories, by = "BGGId")

print("JOIN DATA")
print(nrow(full_games))

# categories list and count categories per game
full_games <- full_games %>% 
  mutate(Categories = apply(select(., 3:10), 1, function(row) paste(names(.)[3:10][row == 1], collapse = ", ")),
         Num_Categories = rowSums(select(., 3:10)))

# subcategories list and count subcategories per game
full_games <- full_games %>% 
  mutate(Sub_Categories = apply(select(., 11:20), 1, function(row) paste(names(.)[11:20][row == 1], collapse = ", ")),
         Num_Sub_Categories = rowSums(select(., 11:20)))

# filter games that are not in any category
full_games <- filter(full_games, Num_Categories > 0)

print("CAT QUANT FILTER")
print(nrow(full_games))

# extract unique categories and print
unique_categories <- sort(unique(unlist(strsplit(na.omit(full_games$Categories), ", "))))
cat("Categories:\n")
cat(paste0(seq_along(unique_categories), ". ", unique_categories), sep = "\n")

# allow user to select categories of interest
selected_categories <- as.integer(strsplit(readline("Select categories of interest by number(s) separated by a space: "), " ")[[1]])
category_names <- unique_categories[selected_categories]
print("You Have Selected:")
print(category_names)
filtered_games <- full_games %>% filter(if_any(all_of(category_names), ~ . == 1))

print("CAT CHOICE FILTER")
print(nrow(filtered_games))

# allow user to select subcategories of interest
y_n <- readline("Would you like to select a subcategory? (Y/N) ")
if (tolower(y_n) == "y") {
  unique_sub_categories <- sort(unique(unlist(strsplit(na.omit(filtered_games$Sub_Categories), ", "))))
  cat("Subcategories:\n")
  cat(paste0(seq_along(unique_sub_categories), ". ", unique_sub_categories), sep = "\n")
  selected_sub_categories <- as.integer(strsplit(readline("Select subcategories by number(s) separated by a space: "), " ")[[1]])
  sub_category_names <- unique_sub_categories[selected_sub_categories]
  print("You Have Selected:")
  print(sub_category_names)
  filtered_games <- filtered_games %>% filter(if_any(all_of(sub_category_names), ~ . == 1))
}

print("SUBCAT CHOICE FILTER")
print(nrow(filtered_games))

# reset index
filtered_games <- filtered_games %>% mutate(Game_Number = row_number())

# display games in a new window
#print(filtered_games %>% select(Game_Number, Name))
View(filtered_games %>% select(Game_Number, BGGId, Name))

# user gives numbers of favorite games
fav_numbers <- as.integer(strsplit(readline("Enter favorite game numbers separated by a space: "), " ")[[1]])
fav_games <- filtered_games %>% filter(Game_Number %in% fav_numbers)

# user gives numbers of least favorite games
least_fav_numbers <- as.integer(strsplit(readline("Enter least favorite game numbers separated by a space: "), " ")[[1]])
least_fav_games <- filtered_games %>% filter(Game_Number %in% least_fav_numbers)

# append new ratings from input to ratings data
new_ratings <- data.frame(
  BGGId = c(fav_games$BGGId, least_fav_games$BGGId),
  Rating = c(rep(10, nrow(fav_games)), rep(1, nrow(least_fav_games))),
  Username = rep("TestUser", nrow(fav_games) + nrow(least_fav_games))
)

ratings <- bind_rows(ratings, new_ratings)

# filter rating data
ratings <- ratings %>% filter(BGGId %in% filtered_games$BGGId)

################# Start collab filtering ################# 
# Assuming your ratings dataframe has columns: Username, BGGId, Rating
ratings_matrix <- ratings %>%
  select(Username, BGGId, Rating) %>%
  spread(key = BGGId, value = Rating, fill = NA) %>%
  column_to_rownames("Username") %>%
  as.matrix()

# Convert the matrix to a sparse matrix
ratings_sparse <- as(ratings_matrix, "CsparseMatrix")
