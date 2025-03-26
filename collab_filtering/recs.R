# DS4420 Final Project
# Victoria Schaller

# import libraries
library(dplyr)
library(tidyr)
library(readr)

# read in data
subcategories <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/subcategories.csv", show_col_types = FALSE)
games <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/games.csv", show_col_types = FALSE)
ratings <- read_csv("C:/Users/toris/Downloads/Spring 25/DS4420/Final Project/game_data/user_ratings.csv", show_col_types = FALSE)

################# User interaction steps ################# 

# rename category columns
colnames(games)[41:48] <- gsub("Cat:", "", colnames(games)[41:48])

# filter columns to id, name, and category data
games <- games %>% select(1, 2, 41:48)

# join games and subcategory data
full_games <- inner_join(games, subcategories, by = "BGGId")

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

# extract unqiue categories and print
unique_categories <- sort(unique(unlist(strsplit(na.omit(full_games$Categories), ", "))))
cat("Categories:\n")
cat(paste0(seq_along(unique_categories), ". ", unique_categories), sep = "\n")

# allow user to select categories of interest
selected_categories <- as.integer(strsplit(readline("Select categories of interest by number(s) separated by a space: "), " ")[[1]])
category_names <- unique_categories[selected_categories]
print("You Have Selected:")
print(category_names)
filtered_games <- full_games %>% filter_at(vars(all_of(category_names)), any_vars(. == 1))

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
  filtered_games <- filtered_games %>% filter_at(vars(sub_category_names), any_vars(. == 1))
}

# reset index
filtered_games <- filtered_games %>% mutate(Game_Number = row_number())

# display games in a new window
#print(filtered_games %>% select(Game_Number, Name))
View(filtered_games %>% select(Game_Number, Name))

# user gives numbers of favorite games
fav_numbers <- as.integer(strsplit(readline("Enter favorite game numbers separated by a space: "), " ")[[1]])
fav_games <- filtered_games %>% filter(Game_Number %in% fav_numbers)

# user gives numbers of least favorite games
least_fav_numbers <- as.integer(strsplit(readline("Enter least favorite game numbers separated by a space: "), " ")[[1]])
least_fav_games <- filtered_games %>% filter(Game_Number %in% least_fav_numbers)

# append new ratings from input to ratings data
new_ratings <- data.frame(
  BGGId = c(fav_games$BGGId, least_fav_games$BGGId),
  Rating = c(rep(5, nrow(fav_games)), rep(1, nrow(least_fav_games))),
  Username = rep("TestUser", nrow(fav_games) + nrow(least_fav_games))
)

ratings <- bind_rows(ratings, new_ratings)

# filter rating data if requested by user
y_n <- readline("Would you like recommendations for only the selected categories? (Y/N) ")
if (tolower(y_n) == "y") {
  ratings <- ratings %>% filter(BGGId %in% filtered_games$BGGId)
}

################# Start collab filtering ################# 

# create ratings matrix
rating_matrix <- ratings %>% 
  pivot_wider(names_from = BGGId, values_from = Rating, values_fn = list(Rating = mean))

# center ratings on item


# create similarity matrix


# 
