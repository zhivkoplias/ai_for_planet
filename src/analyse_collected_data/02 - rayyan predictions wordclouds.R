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

df_global <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/040125-unlabeled/processed/from_rayyan_split/1-global/include/articles.csv')
df_global$extracted_labels <- 'global'
df_earth <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/040125-unlabeled/processed/from_rayyan_split/2-earth/include/articles.csv')
df_earth$extracted_labels <- 'earth'
df_ocean <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/040125-unlabeled/processed/from_rayyan_split/3-ocean/include/articles.csv')
df_ocean$extracted_labels <- 'ocean'
df_water <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/040125-unlabeled/processed/from_rayyan_split/4-water/include/articles.csv')
df_water$extracted_labels <- 'water'
df_biodiversity <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/040125-unlabeled/processed/from_rayyan_split/5-biodiversity/include/articles.csv')
df_biodiversity$extracted_labels <- 'biodiversity'
df_urban <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/040125-unlabeled/processed/from_rayyan_split/6-urban/include/articles.csv')
df_urban$extracted_labels <- 'urban'
df_comms <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/040125-unlabeled/processed/from_rayyan_split/7-comms/include/articles.csv')
df_comms$extracted_labels <- 'comms'
df_policy <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/040125-unlabeled/processed/from_rayyan_split/8-policy/include/articles.csv')
df_policy$extracted_labels <- 'policy'

combined_df <- bind_rows(df_global, df_earth, df_ocean, df_water, df_biodiversity,
                         df_urban, df_comms, df_policy)

combined_df <- combined_df %>%  
  select(doi, extracted_labels, title, abstract)

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
  # mutate(title_words = lemmatize_strings(title_words)) %>%  
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
stopwords <- c("and", "in", "a", "the", "this", "is", "to", "of", 'download', 'occurrence', 'Download', 'Occurrence')
stopwords <- stopwords::stopwords("en") 


# Apply the function to aggregate word counts  
aggregated_data_for_wordclouds <- aggregate_word_counts(combined_df, stopwords)

# Convert the data frame to a matrix  
aggregated_data_for_wordclouds_tdm <- as.matrix(aggregated_data_for_wordclouds[, -1])  # Exclude the first column (Terms)  
rownames(aggregated_data_for_wordclouds_tdm) <- aggregated_data_for_wordclouds$unique_word  


#Plot comparative word clouds (set 1)
# Define the color palette  
color_palette <- c("#E63946", "#A8DADC", "#457B9D",   
                   "#1D3557", "#F1C40F", "#F77F00", "#FABD60",   
                   "#2A9D8F", "#E9C46A")

svg("../plots/new/predictions_wordcloud_plot.svg", width=10, height=10)  # Open SVG device; adjust size as needed  

par(mfrow=c(1,1))  
comparison.cloud(aggregated_data_for_wordclouds_tdm, random.order=FALSE, colors = color_palette,  
                 title.size=2.5, max.words=400)  

dev.off()  # Close the SVG device and save the file  





#Plot category counts (plot 2)
# Count unique rows for each category in extracted_labels  
category_counts <- combined_df %>%  
  group_by(extracted_labels) %>%  
  summarise(unique_count = n_distinct(doi), .groups = 'drop') %>%  
  arrange(desc(unique_count))  # Sort by count in descending order  

category_counts <- combined_df %>%  
  # Create a new column with the first label only  
  mutate(first_label = str_split(extracted_labels, ",\\s*")) %>%  
  mutate(first_label = sapply(first_label, `[`, 1)) %>%  # Extract the first label  
  group_by(first_label) %>%  # Group by the first label  
  summarise(unique_count = n_distinct(doi), .groups = 'drop') %>%  
  arrange(desc(unique_count))  # Sort by count in descending order  

svg("../plots/new/predictions_histogram.svg", width=10, height=10)  # Open SVG device; adjust size as needed  
# Create the histogram  
ggplot(category_counts, aes(x = reorder(first_label, -unique_count), y = unique_count)) +  
  geom_bar(stat = "identity", fill = "steelblue") +  
  labs(title = "Predictions",  
       x = "Issue areas",  
       y = "Number of articles") +  
  theme_minimal(base_size = 15) +  # Increase base font size for better visibility  
  theme(axis.text.x = element_text(angle = 45, hjust = 1))  # Rotate x-axis labels for better readability 
dev.off()  # Close the SVG device and save the file  