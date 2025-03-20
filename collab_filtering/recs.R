# Sample DataFrame
df <- data.frame(Names = c("Alice", "Bob", "Charlie", "David"))

# Create a numeric menu for selection
if (interactive()) {
  selected_index <- menu(df$Names, title = "Select a number corresponding to a name:")
  
  if (selected_index > 0) {
    cat("You have chosen", df$Names[selected_index], "\n")
  } else {
    cat("No selection made.\n")
  }
} else {
  cat("This script must be run interactively.\n")
}



