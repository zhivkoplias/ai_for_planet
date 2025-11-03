library(wordcloud)
#library(tdm)
library(stringr)
library(dplyr)
library(tidyr)
library(textstem)
library(tm)
library(SnowballC)  
library(ggplot2)
library(stopwords)
setwd("~/SRC/AI/Deep_mind_roundtable/Lit search/scripts")

data_word_clouds <- read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/090125_all_included/articles.csv')

# Function to extract labels from a given text  
extract_labels <- function(text) {  
  # Define the regex pattern  
  pattern <- "RAYYAN-LABELS: (.*?)(\\||$)"  
  
  # Use str_extract to find the labels  
  labels <- str_extract(text, pattern)  
  
  # Clean up the result to return only the labels  
  if (!is.na(labels)) {  
    labels <- str_remove(labels, "RAYYAN-LABELS: ")  
    labels <- str_remove(labels, "\\|$")  
    return(labels)  
  } else {  
    return(NA)  # Return NA if no match is found  
  }  
}  

# Apply the function to the dataframe  
data_word_clouds_w_labels <- data_word_clouds %>%  
  mutate(extracted_labels = sapply(notes, extract_labels))

# Function to filter labels based on approved list  
filter_labels <- function(df, approved_labels) {  
  # Ensure approved_labels is a character vector  
  approved_labels <- as.character(approved_labels)  
  
  # Function to filter labels in each row  
  filter_row_labels <- function(labels) {  
    # Split the labels by comma and trim whitespace  
    if (!is.na(labels)) {  
      label_list <- unlist(str_split(labels, ","))  # Split by comma  
      label_list <- str_trim(label_list)  # Trim whitespace from each label  
      filtered_labels <- label_list[label_list %in% approved_labels]  # Keep only approved labels  
      return(paste(filtered_labels, collapse = ", "))  # Return as a single string  
    } else {  
      return(NA)  # Return NA if no labels  
    }  
  }  
  
  # Apply the filtering function to the extracted_labels column  
  df <- df %>%  
    mutate(extracted_labels = sapply(extracted_labels, filter_row_labels))  
  
  return(df)  # Return the updated dataframe  
}   

# List of approved labels  
approved_labels <- c("anthropocene", "earth", "climate", "ocean", "water", "biodiversity", "comms", "policy", "urban")  

# Apply the function to filter the dataframe  
data_word_clouds_w_labels <- filter_labels(data_word_clouds_w_labels, approved_labels)

#create training set ids
#topic 1
article_ids_df_anthropocene <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'anthropocene') %>%  
  select(key)
article_ids_df_climate <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'climate') %>%  
  select(key)
article_ids_df_global <- rbind(article_ids_df_anthropocene, article_ids_df_climate)

#topic 2
article_ids_df_earth <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'earth') %>%  
  select(key)

#topic3
article_ids_df_ocean <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'ocean') %>%  
  select(key)

#topic4
article_ids_df_water <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'water') %>%  
  select(key)

#topic5
article_ids_df_biodiversity <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'biodiversity') %>%  
  select(key)

#topic6
article_ids_df_comms <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'comms') %>%  
  select(key)

#topic7
article_ids_df_policy <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'policy') %>%  
  select(key)

#topic8
article_ids_df_urban <- data_word_clouds_w_labels %>%  
  filter(extracted_labels == 'urban') %>%  
  select(key)

###extract the actual papers and save it as csv file
df_to_sort_included_articles_topic1_global <- df_to_sort_included_articles %>%  
  filter(key %in% article_ids_df_global$key)
# Save the filtered dataframe to a CSV file  
write.csv(df_to_sort_included_articles_topic1_global, "../training_set_split8/final_topics_included/topic1_global_included.csv", row.names = FALSE)  

df_to_sort_included_articles_topic2_earth <- df_to_sort_included_articles %>%  
  filter(key %in% article_ids_df_earth$key)
# Save the filtered dataframe to a CSV file  
write.csv(df_to_sort_included_articles_topic2_earth, "../training_set_split8/final_topics_included/topic2_earth_included.csv", row.names = FALSE)  

df_to_sort_included_articles_topic3_ocean <- df_to_sort_included_articles %>%  
  filter(key %in% article_ids_df_ocean$key)
# Save the filtered dataframe to a CSV file  
write.csv(df_to_sort_included_articles_topic3_ocean, "../training_set_split8/final_topics_included/topic3_ocean_included.csv", row.names = FALSE)  

df_to_sort_included_articles_topic4_water <- df_to_sort_included_articles %>%  
  filter(key %in% article_ids_df_water$key)
# Save the filtered dataframe to a CSV file  
write.csv(df_to_sort_included_articles_topic4_water, "../training_set_split8/final_topics_included/topic4_water_included.csv", row.names = FALSE)  

df_to_sort_included_articles_topic5_biodiversity <- df_to_sort_included_articles %>%  
  filter(key %in% article_ids_df_biodiversity$key)
# Save the filtered dataframe to a CSV file  
write.csv(df_to_sort_included_articles_topic5_biodiversity, "../training_set_split8/final_topics_included/topic5_biodiversity_included.csv", row.names = FALSE)  

df_to_sort_included_articles_topic6_comms <- df_to_sort_included_articles %>%  
  filter(key %in% article_ids_df_comms$key)
# Save the filtered dataframe to a CSV file  
write.csv(df_to_sort_included_articles_topic6_comms, "../training_set_split8/final_topics_included/topic6_comms_included.csv", row.names = FALSE)  

df_to_sort_included_articles_topic7_policy <- df_to_sort_included_articles %>%  
  filter(key %in% article_ids_df_policy$key)
# Save the filtered dataframe to a CSV file  
write.csv(df_to_sort_included_articles_topic7_policy, "../training_set_split8/final_topics_included/topic7_policy_included.csv", row.names = FALSE)  

df_to_sort_included_articles_topic8_urban <- df_to_sort_included_articles %>%  
  filter(key %in% article_ids_df_urban$key)
# Save the filtered dataframe to a CSV file  
write.csv(df_to_sort_included_articles_topic8_urban, "../training_set_split8/final_topics_included/topic8_urban_included.csv", row.names = FALSE)  

#rename for consistency
data_word_clouds_w_labels <- data_word_clouds_w_labels %>%  
  mutate(extracted_labels = if_else(extracted_labels %in% c('anthropocene', 'climate'),   
                                    'global',   
                                    extracted_labels))  


##WORDCLOUDS
data_word_clouds_w_labels <- data_word_clouds_w_labels %>%  
  select(doi, extracted_labels, title, abstract)

data_word_clouds_w_labels_nona <- data_word_clouds_w_labels %>%  
  filter(!is.na(extracted_labels) & trimws(extracted_labels) != "")

# Function to aggregate data by categories and calculate word counts  
aggregate_word_counts <- function(df, stopwords = NULL) {  
  # Ensure the extracted_labels column is treated as a character vector  
  df$extracted_labels <- as.character(df$extracted_labels)  
  
  # Split the extracted_labels into separate rows  
  df_long <- df %>%  
    mutate(extracted_labels = str_split(extracted_labels, ",\\s*")) %>%  
    unnest(extracted_labels)  # Expand the dataframe  
  
  # Count words in titles  
  word_counts <- df_long %>%  
    mutate(title_words = str_split(title, "\\s+")) %>%  
    unnest(title_words) %>%  
    # De-capitalize words  
    mutate(title_words = tolower(title_words)) %>%  # Convert to lowercase  
    # Lemmatize words using the textstem package  
  #  mutate(title_words = lemmatize_strings(title_words)) %>%  
    # Stem words using the SnowballC package  
  #  mutate(title_words = wordStem(title_words)) %>%  
    # Remove stopwords if provided  
    filter(is.null(stopwords) | !title_words %in% stopwords) %>%  
    group_by(extracted_labels, title_words) %>%  
    summarise(count = n(), .groups = 'drop') %>%  
    filter(count >= 2) %>%  # Keep only words that appear at least twice  
    pivot_wider(names_from = extracted_labels, values_from = count, values_fill = list(count = 0))  # Pivot to wide format  
  
  # Create a final dataframe with unique words and their counts across categories  
  final_df <- word_counts %>%  
    select(title_words, everything()) %>%  
    rename(unique_word = title_words) %>%  
    arrange(unique_word)  # Sort by unique words  
  
  return(final_df)  # Return the final dataframe  
}  

# Define stopwords  
stopwords <- c("and", "in", "a", "the", "this", "is", "to", "of")  


# Apply the function to aggregate word counts  
aggregated_data_for_wordclouds <- aggregate_word_counts(data_word_clouds_w_labels_nona, stopwords)
#colnames(aggregated_data_for_wordclouds) <- c("unique_word", "global", "biodiversity", "climate", "comms", "earth", "ocean", "decisions", "urban", "water")
colnames(aggregated_data_for_wordclouds) <- c("unique_word", "biodiversity", "comms", "earth", "global", "ocean", "policy", "urban", "water")
# Convert the data frame to a matrix  
aggregated_data_for_wordclouds_tdm <- as.matrix(aggregated_data_for_wordclouds[, -1])  # Exclude the first column (Terms)  
rownames(aggregated_data_for_wordclouds_tdm) <- aggregated_data_for_wordclouds$unique_word  

#Plot comparative word clouds (set 1)

# Define the color palette  
color_palette <- c("#E63946", "#A8DADC", "#457B9D",   
                   "#1D3557", "#F1C40F", "#F77F00", "#FABD60",   
                   "#2A9D8F", "#E9C46A")

svg("../plots/new/training_set_wordcloud_plot.svg", width=10, height=10)  # Open SVG device; adjust size as needed  

par(mfrow=c(1,1))  

comparison.cloud(aggregated_data_for_wordclouds_tdm, random.order=FALSE, colors = color_palette,
                 title.size=2.5, max.words=400)
dev.off()  # Close the SVG device and save the file  

#Plot category counts (plot 2)
# Count unique rows for each category in extracted_labels  
category_counts <- data_word_clouds_w_labels_nona %>%  
  group_by(extracted_labels) %>%  
  summarise(unique_count = n_distinct(doi), .groups = 'drop') %>%  
  arrange(desc(unique_count))  # Sort by count in descending order  

category_counts <- data_word_clouds_w_labels_nona %>%  
  # Create a new column with the first label only  
  mutate(first_label = str_split(extracted_labels, ",\\s*")) %>%  
  mutate(first_label = sapply(first_label, `[`, 1)) %>%  # Extract the first label  
  group_by(first_label) %>%  # Group by the first label  
  summarise(unique_count = n_distinct(doi), .groups = 'drop') %>%  
  arrange(desc(unique_count))  # Sort by count in descending order  

svg("../plots/new/training_set_histogram.svg", width=10, height=10)  # Open SVG device; adjust size as needed  
# Create the histogram  
ggplot(category_counts, aes(x = reorder(first_label, -unique_count), y = unique_count)) +  
  geom_bar(stat = "identity", fill = "steelblue") +  
  labs(title = "Training set",  
       x = "Topics",  
       y = "Number of selected articles") +  
  theme_minimal(base_size = 15) +  # Increase base font size for better visibility
 # Close the SVG device and save the file  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability  
dev.off() 