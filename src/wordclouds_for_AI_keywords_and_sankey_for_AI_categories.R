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
setwd("/home/erikz/SRC/research/llm_classification_experiments/src")

data_word_clouds <-read.csv('../070125-labeled/exported_from_rayyan/articles.csv')
data_word_clouds <-read.csv('../090125-alldata-raayan/training_set/090125_all_included/articles.csv')
data_word_clouds <- read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/090125_all_excluded_and_included/articles.csv')
df_to_sort_included_articles <- read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/090125_all_included/articles.csv')
df_with_extracted_keywords <- read.csv('../llm_output/AI_with_keywords_filtered.out')

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
    mutate(AI_gen = sapply(AI_gen, filter_row_labels))  
  
  return(df)  # Return the updated dataframe  
}   

# List of approved labels  
approved_labels <- c("anthropocene", "earth", "climate", "ocean", "water", "biodiversity", "comms", "policy", "urban")
approved_labels <- c('Classic Machine Learning', 'Classic AI & Neural Network Architectures',
                     'Classic Deep Learning', 'New Generation of AI')

# Apply the function to filter the dataframe  
data_word_clouds_w_labels <- filter_labels(df_with_extracted_keywords, approved_labels)

# Preprocess the text  
data_word_clouds_w_labels_cleaned <- data_word_clouds_w_labels %>%  
  mutate(keywords_preprocessed = tolower(keywords_preprocessed),                        # Convert to lowercase  
         keywords_preprocessed = gsub("'", "", keywords_preprocessed),                  # Remove single quotes  
         keywords_preprocessed = gsub("\\s+", " ", keywords_preprocessed),              # Replace multiple whitespace with a single space  
         keywords_preprocessed = gsub("[\\[\\]()]", "", keywords_preprocessed),         # Remove brackets and parentheses  
         keywords_preprocessed = trimws(keywords_preprocessed)) %>%                     # Trim whitespace  
  filter(keywords_preprocessed != "")                                                   # Remove empty strings  

# Perform stemming on the cleaned text  
data_word_clouds_w_labels_cleaned$keywords_preprocessed_stem <- textstem::stem_words(data_word_clouds_w_labels_cleaned$keywords_preprocessed)

#rename
data_word_clouds_w_labels_cleaned <- data_word_clouds_w_labels_cleaned %>%  
  mutate(AI_gen = ifelse(AI_gen == 'Classic AI & Neural Network Architectures',   
                         'Classic Deep Learning',   
                         AI_gen))  

#select AI gens
data_word_clouds_w_labels_cleaned_selected <- data_word_clouds_w_labels_cleaned %>%  
  select(doi, AI_gen, keywords_preprocessed, keywords_preprocessed_stem)

data_word_clouds_w_labels_cleaned_selected_nona <- data_word_clouds_w_labels_cleaned_selected %>%  
  filter(!is.na(AI_gen) & trimws(AI_gen) != "")

article_ids_df_ml <- data_word_clouds_w_labels_cleaned_selected_nona %>%  
  filter(AI_gen == 'Classic Machine Learning')

article_ids_df_dl <- data_word_clouds_w_labels_cleaned_selected_nona %>%  
  filter(AI_gen == 'Classic Deep Learning') 

article_ids_df_new <- data_word_clouds_w_labels_cleaned_selected_nona %>%  
  filter(AI_gen == 'New Generation of AI')


aggregate_word_counts <- function(df, stopwords = NULL) {  
  # Ensure the extracted_labels column is treated as a character vector  
  df$AI_gen <- as.character(df$AI_gen)  
  
  # Split the extracted_labels into separate rows  
  df_long <- df %>%  
    mutate(AI_gen = str_split(AI_gen, ",\\s*")) %>%  
    unnest(AI_gen)  # Expand the dataframe  
  
  # Count words in titles  
  word_counts <- df_long %>%  
    mutate(title_words = str_split(keywords_preprocessed, "\\s+")) %>%  
    unnest(title_words) %>%  
    # De-capitalize words  
    mutate(title_words = tolower(title_words)) %>%  # Convert to lowercase  
    # Lemmatize words using the textstem package  
    #  mutate(title_words = lemmatize_strings(title_words)) %>%  
    # Stem words using the SnowballC package  
    #  mutate(title_words = wordStem(title_words)) %>%  
    # Remove stopwords if provided  
    filter(is.null(stopwords) | !title_words %in% stopwords) %>%  
    group_by(AI_gen, title_words) %>%  
    summarise(count = n(), .groups = 'drop') %>%  
    filter(count >= 2) %>%  # Keep only words that appear at least twice  
    pivot_wider(names_from = AI_gen, values_from = count, values_fill = list(count = 0))  # Pivot to wide format  
  
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
aggregated_data_for_wordclouds <- aggregate_word_counts(data_word_clouds_w_labels_cleaned_selected_nona, stopwords)
colnames(aggregated_data_for_wordclouds) <- c("unique_word", "classic DL", "classic ML", "new generation of AI")

# Convert the data frame to a matrix  
aggregated_data_for_wordclouds_tdm <- as.matrix(aggregated_data_for_wordclouds[, -1])  # Exclude the first column (Terms)  
rownames(aggregated_data_for_wordclouds_tdm) <- aggregated_data_for_wordclouds$unique_word  

#Plot comparative word clouds (set 1)

# Define the color palette  
color_palette <- c("#E63946", "#A8DADC", "#457B9D")


# Set your SVG output file name  
svg("wordcloud_plot.svg", width = 10, height = 6)  # You can adjust width and height as needed  

# Set up the plotting layout  
par(mfrow=c(1,1))  

# Create the comparison cloud  
comparison.cloud(aggregated_data_for_wordclouds_tdm,  
                 random.order=FALSE,   
                 colors = color_palette,  
                 title.size=2.0,   
                 max.words=800)  

# Close the SVG device  
dev.off()  

#select training set for sankey
data_word_clouds_w_labels_cleaned_selected_nona$issue_area <- NA
topic1_global <- read.csv('../data/topic1_global_included.csv')
topic2_earth <- read.csv('../data/topic2_earth_included.csv')
topic3_ocean <- read.csv('../data/topic3_ocean_included.csv')
topic4_water <- read.csv('../data/topic4_water_included.csv')
topic5_biodiversity <- read.csv('../data/topic5_biodiversity_included.csv')
topic6_comms <- read.csv('../data/topic6_comms_included.csv')
topic7_policy <- read.csv('../data/topic7_policy_included.csv')
topic8_urban <- read.csv('../data/topic8_urban_included.csv')

# Get a list of all topic data frames (names of objects starting with "topic")  
topic_dfs <- grep("^topic", ls(), value = TRUE)  

# Iterate over each data frame and check for doi matches  
for (topic in topic_dfs) {  
  # Get the current topic data frame as a variable  
  topic_data <- get(topic)  
  
  # Ensure "doi" column exists in the topic data frame  
  if ("doi" %in% colnames(topic_data)) {  
    # Check where the "doi" in the main data frame matches the current topic data frame  
    matched_rows <- data_word_clouds_w_labels_cleaned_selected_nona$doi %in% topic_data$doi  
    
    # Update the `issue_area` column with the name of the data frame where a match is found  
    data_word_clouds_w_labels_cleaned_selected_nona$issue_area[matched_rows] <- topic  
  }  
}  
data_word_clouds_w_labels_cleaned_selected_nona_with_doi <- data_word_clouds_w_labels_cleaned_selected_nona[!is.na(data_word_clouds_w_labels_cleaned_selected_nona$doi), ] 
df_training_data_with_AI_keywords <- data_word_clouds_w_labels_cleaned_selected_nona_with_doi[!is.na(data_word_clouds_w_labels_cleaned_selected_nona_with_doi$issue_area), ]  

df_training_data_with_AI_keywords <- df_training_data_with_AI_keywords[  
  !is.na(df_training_data_with_AI_keywords$doi) &   
    df_training_data_with_AI_keywords$doi != "",   
]  

# Aggregate the data to calculate weights (occurrences of each AI_gen and issue_area combination)  
df_training_data_with_AI_keywords_aggregated <- df_training_data_with_AI_keywords %>%  
  dplyr::count(AI_gen, issue_area, name = "weight")  

### plot sankey diagram
# Load necessary library  
library(networkD3)
library(htmlwidgets)
library(webshot2)
library(rsvg)

# Create nodes: unique categories from both columns  
nodes <- data.frame(name = unique(c(df_training_data_with_AI_keywords_aggregated$AI_gen,   
                                    df_training_data_with_AI_keywords_aggregated$issue_area)),  
                    stringsAsFactors = FALSE)  

# Map source and target to indices in nodes  
df_training_data_with_AI_keywords_aggregated$source <- match(df_training_data_with_AI_keywords_aggregated$AI_gen, nodes$name) - 1  
df_training_data_with_AI_keywords_aggregated$target <- match(df_training_data_with_AI_keywords_aggregated$issue_area, nodes$name) - 1  

# Create links data frame (source, target, and value are required)  
links <- data.frame(  
  source = df_training_data_with_AI_keywords_aggregated$source,   
  target = df_training_data_with_AI_keywords_aggregated$target,   
  value = df_training_data_with_AI_keywords_aggregated$weight  
)  

# Define grey color for all nodes
node_colors <- 'd3.scaleOrdinal()
    .domain(["Classic AI", "Machine Learning", "Deep Learning",
             "topic1_global", "topic2_earth", "topic3_ocean",
             "topic4_water", "topic5_biodiversity", "topic6_comms",
             "topic7_policy", "topic8_urban"])
    .range(["#808080"])'  # Standard grey color

# Create the Sankey diagram with grey nodes
sankey_plot <- sankeyNetwork(
  Links = links, 
  Nodes = nodes,
  Source = "source",
  Target = "target",
  Value = "value",
  NodeID = "name",
  units = "TWh",
  fontSize = 28,
  nodeWidth = 30,
  sinksRight = TRUE,
  colourScale = node_colors  # Add this line to apply colors
)

# Save as an SVG file using `saveWidget` with self-contained = FALSE  
saveWidget(sankey_plot, "sankey_diagram_AI_gens.html")
webshot("sankey_diagram_AI_gens.html", "sankey_diagram_AI_gens.pdf", delay=1)
rsvg_pdf("sankey_diagram_AI_gens.pdf", "sankey_diagram_AI_gens.svg")      # Convert PDF to SVG
