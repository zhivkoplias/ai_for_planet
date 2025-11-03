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
setwd("/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/scripts")

data_word_clouds <-read.csv('../070125-labeled/exported_from_rayyan/articles.csv')
data_word_clouds <-read.csv('../090125-alldata-raayan/training_set/090125_all_included/articles.csv')
data_word_clouds <- read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/090125_all_excluded_and_included/articles.csv')
df_to_sort_included_articles <- read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/090125_all_included/articles.csv')
df_with_extracted_keywords <- read.csv('/home/erikz/SRC/research/llm_classification_experiments/llm_output/AI_with_keywords_filtered.out')
df_with_extracted_keywords <- read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/classification-with-llm/df_with_generations_for_wordclouds.csv')

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
approved_labels <- c('Classic ML', 'Classic DL', 'New AI')

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

#select AI gens
data_word_clouds_w_labels_cleaned_selected <- data_word_clouds_w_labels_cleaned %>%  
  select(doi, AI_gen, keywords_preprocessed, keywords_preprocessed_stem)

data_word_clouds_w_labels_cleaned_selected_nona <- data_word_clouds_w_labels_cleaned_selected %>%  
  filter(!is.na(AI_gen) & trimws(AI_gen) != "")

article_ids_df_ml <- data_word_clouds_w_labels_cleaned_selected_nona %>%  
  filter(AI_gen == 'Classic ML')

article_ids_df_dl <- data_word_clouds_w_labels_cleaned_selected_nona %>%  
  filter(AI_gen == 'Classic DL') 

article_ids_df_new <- data_word_clouds_w_labels_cleaned_selected_nona %>%  
  filter(AI_gen == 'New AI')


aggregate_word_counts <- function(df, stopwords = NULL, method_mapping = NULL) {  
  library(dplyr)  
  library(tidyr)  
  library(stringr)  
  
  # Ensure AI_gen is character vector  
  df$AI_gen <- as.character(df$AI_gen)  
  
  # Split AI_gen into separate rows  
  df_long <- df %>%  
    mutate(AI_gen = str_split(AI_gen, ",\\s*")) %>%  
    unnest(AI_gen)  
  
  # Count words and map to canonical forms  
  word_counts <- df_long %>%  
    mutate(title_words = str_split(keywords_preprocessed, "\\s+")) %>%  
    unnest(title_words) %>%  
    mutate(title_words = tolower(title_words)) %>%  
    filter(is.null(stopwords) | !title_words %in% stopwords) %>%  
    mutate(  
      canonical_word = if (!is.null(method_mapping)) {  
        ifelse(  
          title_words %in% names(method_mapping),  
          unname(method_mapping[title_words]),  
          title_words  
        )  
      } else {  
        title_words  
      }  
    ) %>%  
    group_by(AI_gen, canonical_word) %>%  
    summarise(count = n(), .groups = 'drop') %>%  
    filter(count >= 2) %>%  
    pivot_wider(names_from = AI_gen, values_from = count, values_fill = list(count = 0))  
  
  # Final dataframe: unique canonical words and their counts across AI_gen  
  final_df <- word_counts %>%  
    select(canonical_word, everything()) %>%  
    rename(unique_word = canonical_word) %>%  
    arrange(unique_word)  
  
  return(final_df)  
}  

method_mapping_vec <- c(  
  # Machine Learning & AI Core Methods  
  'neural network' = 'neural network',  
  'neural networks' = 'neural network',  
  'cnn' = 'convolutional neural network',  
  'convolutional neural network' = 'convolutional neural network',  
  'deep learning' = 'deep learning',  
  'machine learning' = 'machine learning',  
  'machine learn' = 'machine learning',  
  'artificial intelligence ai' = 'artificial intelligence',  
  'artificial intelligence' = 'artificial intelligence',  
  'advanced ai technique' = 'artificial intelligence',  
  'advanced ai techniques' = 'artificial intelligence',  
  'ai learning' = 'artificial intelligence',  
  'ai technology' = 'artificial intelligence',  
  'applied ai' = 'artificial intelligence',  
  'generative ai' = 'generative ai',  
  'reinforcement learn' = 'reinforcement learning',  
  'reinforcement learning' = 'reinforcement learning',  
  'q-learning' = 'reinforcement learning',  
  'qlearning' = 'reinforcement learning',  
  'foundation model' = 'foundation model',  
  'transformers' = 'transformer',  
  'svm' = 'support vector machine',  
  'svms' = 'support vector machine',  
  'support vector machine' = 'support vector machine',  
  'decision tree' = 'decision tree',  
  'decisionmaking' = 'decision making',  
  'decisionmaking process' = 'decision making',  
  'decision support system' = 'decision support system',  
  'classification' = 'classification',  
  'regression' = 'regression',  
  'linear regression' = 'linear regression',  
  'logistic regression' = 'logistic regression',  
  'predictive analysis' = 'predictive analytics',  
  'sentiment analysis' = 'sentiment analysis',  
  'text mining' = 'text mining',  
  'data mining' = 'data mining',  
  'data mine' = 'data mining',  
  'analytics' = 'analytics',  
  'big data analytics' = 'big data analytics',  
  'big data analysis' = 'big data analytics',  
  'big data' = 'big data analytics',  
  'data analysis' = 'data analytics',  
  'data analysis univariate multivariate' = 'data analytics',  
  'data visualization' = 'data visualization',  
  'simulation' = 'simulation',  
  'content analysis' = 'content analysis',  
  # Spatial & Remote Sensing  
  'gis' = 'gis',  
  'gis tool' = 'gis',  
  'gis sense' = 'remote sensing',  
  'remote sense' = 'remote sensing',  
  'remote sense data station' = 'remote sensing',  
  'ndvi' = 'ndvi',  
  'earth observation' = 'remote sensing',  
  # Robotics & IoT  
  'iot' = 'iot',  
  'iot device' = 'iot',  
  'robotics' = 'robotics',  
  'drones' = 'drones',  
  # Blockchain & Cyber  
  'blockchain' = 'blockchain',  
  'blockchainbased artificial intelligence' = 'blockchain',  
  'cybersecurity' = 'cybersecurity',  
  # Modelling Approaches  
  'agentbased modelling' = 'agent-based modeling',  
  'entropy model' = 'entropy model',  
  'earth system model' = 'earth system model',  
  'highresolution model' = 'high-resolution model',  
  'modeling' = 'modeling',  
  # Statistical and Math  
  'statistical method' = 'statistical method',  
  'fuzzy logic' = 'fuzzy logic',  
  'eof decomposition filtering' = 'eof decomposition filtering',  
  'ahp method' = 'ahp',  
  'euclidean distance' = 'euclidean distance',  
  # Decision and Support  
  'decision making' = 'decision making',  
  'decision support system' = 'decision support system',  
  # Urbanism & Environment  
  'urban planning' = 'urban planning',  
  'sustainable urban planning' = 'urban planning',  
  'urban environment' = 'urban environment',  
  'urban form analysis' = 'urban form analysis',  
  'biodiversity conservation' = 'biodiversity conservation',  
  'ecosystem service value esv' = 'ecosystem service valuation',  
  'global water resource management' = 'water management',  
  'renewable energy' = 'renewable energy',  
  # Misc, Tech & Science  
  'digital twin' = 'digital twin',  
  'virtual reality' = 'virtual reality',  
  'cloud computing' = 'cloud computing',  
  'computer vision' = 'computer vision',  
  'geospatial data' = 'geospatial data',  
  'graph database' = 'graph database',  
  'software architecture' = 'software architecture',  
  'generative ai' = 'generative ai',  
  'minirov' = 'robotics',  
  'text mining' = 'text mining',  
  # AI method aggregation  
  'long shortterm memory lstms' = 'LSTM',  
  'long shortterm memory' = 'LSTM',  
  'long shortterm memory lstm' = 'LSTM',  
  'lstm' = 'LSTM',  
  'forest' = 'random forest',  
  'supervised learn' = 'supervised learning',  
  'anns' = 'ANN',  
  'ann' = 'ANN',  
  'cnn' = 'CNN',  
  'convolutional neural network' = 'CNN',  
  'unet architecture' = 'unet',  
  'gans' = 'GAN',  
  'llms' = 'LLM',  
  'large language model' = 'LLM',  
  'llms gpt4 chatgpt llama' = 'LLM',  
  'multimodal model clip' = 'CLIP',  
  'clip' = 'CLIP',  
  'gpt4' = 'chatgpt',  
  'chatgpt4o' = 'chatgpt',  
  'large language model llm' = 'LLM',  
  'gpt3' = 'chatgpt',  
  'gpt' = 'chatgpt',  
  'generative adversarial network' = 'GAN',  
  'gan' = 'GAN',  
  'generative adversarial network gan' = 'GAN',  
  'generative adversarial network gin' = 'GAN',  
  'chatgpt4' = 'chatgpt'  
)  

# Define stopwords  
stopwords <- c("and", "in", "a", "the", "this", "is", "to", "of")  

# Apply the function to aggregate word counts  
aggregated_data_for_wordclouds <- aggregate_word_counts(data_word_clouds_w_labels_cleaned_selected_nona,
                                                        stopwords, method_mapping_vec)
colnames(aggregated_data_for_wordclouds) <- c("unique_word", "Classic DL", "Classic ML", "New AI")

# Convert the data frame to a matrix  
aggregated_data_for_wordclouds_tdm <- as.matrix(aggregated_data_for_wordclouds[, -1])  # Exclude the first column (Terms)  
rownames(aggregated_data_for_wordclouds_tdm) <- aggregated_data_for_wordclouds$unique_word  

#Plot comparative word clouds (set 1)

# Define the color palette  
color_palette <- c("#E63946", "#A8DADC", "#457B9D")


# Set your SVG output file name  
svg("../plots/new/wordcloud_plot_AI_gens.svg", width = 10, height = 10)  # You can adjust width and height as needed  

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