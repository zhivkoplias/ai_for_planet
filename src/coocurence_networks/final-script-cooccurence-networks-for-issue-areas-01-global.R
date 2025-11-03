library(dplyr)
library(tidyr)
library(scales)
library(tidygraph)
library(ggraph)
library(ggplot2)
library(ggrepel)
library(stringr)  # Added for str_split
library(svglite)  # Added for SVG export
library(purrr)

setwd("/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/lit_review_deepmind/results")

# Read and preprocess data
df <- read.csv('literature_review_fulldata.csv')
df_selected <- df %>%
  distinct(doi, title, .keep_all = TRUE) %>%
  select(article_keywords, AI_keywords_aggregated, issue_area)

colnames(df_selected) <- c("article_keywords", "AI_keywords", "issue_area")

###GLOBAL
save_file_name <- "keyword_network_global_risks.svg"
issue_label <- "global"
plot_title <- "Keyword Network Analysis: Global Risks"

# Define allowed keywords in lowercase for case-insensitive matching
allowed_keywords <- tolower(c(
  "neural network",
  "random forest",
  "CNN",
  "LSTM",
  "support vector machine",
  "regression",
  "unet",
  "transformer",
  "decision tree",
  "machine learning",
  "GAN",
  "supervised learning"
))

# Corrected function with proper syntax
clean_keywords <- function(text) {
  # Handle missing and empty values first
  if (is.na(text)) return(NA_character_)
  if (trimws(text) == "") return("")
  
  # Process non-empty text
  keywords <- strsplit(text, ",\\s*")[[1]]
  trimmed <- trimws(keywords)
  lower_trimmed <- tolower(trimmed)
  
  # Keep only allowed keywords and preserve original casing
  valid <- trimmed[lower_trimmed %in% allowed_keywords]
  paste(valid, collapse = ", ")
}

# Apply cleaning and filtering
df_raw <- df_selected %>%
  filter(issue_area == issue_label) %>%
  mutate(
    AI_keywords_clean = sapply(AI_keywords, clean_keywords)
  ) %>%
  filter(AI_keywords_clean != "") %>%
  mutate(AI_keywords = AI_keywords_clean) %>%
  select(-AI_keywords_clean)


# Step 3: Get the top 12 individual article keywords by frequency
library(tidyverse)

# Split and count keywords
keyword_freq <- df_raw %>%
  mutate(keyword = strsplit(article_keywords, ",\\s*")) %>%
  unnest(keyword) %>%
  mutate(clean_keyword = trimws(tolower(keyword))) %>%
  filter(clean_keyword != "") %>%
  count(clean_keyword, name = "n") %>%
  arrange(desc(n))

# Get the top 12 keywords (in lowercase for matching)
top12_keywords <- keyword_freq %>%
  slice_max(n, n = 12) %>%
  pull(clean_keyword)

# Step 4: Clean article_keywords to keep only top 12 keywords
clean_keyword_column <- function(x) {
  if (is.na(x) || str_trim(x) == "") return("")
  
  # Split, trim, and remove empty strings
  keys <- strsplit(x, ",\\s*")[[1]] %>%
    trimws() %>%
    .[. != ""]
  
  # Keep only keywords in top12 (case-insensitive match)
  keep <- keys[tolower(keys) %in% top12_keywords]
  
  # Return comma-separated string or empty
  paste(keep, collapse = ", ")
}

raw_data <- df_raw %>%
  mutate(
    # Process each article_keywords entry
    article_keywords = sapply(article_keywords, clean_keyword_column)
  ) %>%
  # Remove rows with no keywords left
  filter(article_keywords != "") %>%
  select(article_keywords, AI_keywords, issue_area)

#select top12 article keywords
# Function to calculate top keywords
get_top_keywords <- function(data, column, n = 12) {
  data %>%
    separate_rows({{column}}, sep = ",\\s*") %>%
    count({{column}}, sort = TRUE) %>%
    # First arrange by frequency (desc) and then by keyword (asc) for consistent tie-breaking
    arrange(desc(n), {{column}}) %>%
    # Take exactly n rows
    head(n) %>%
    pull({{column}})
}

# Get top 12 article_keywords
top_article_keywords <- get_top_keywords(raw_data, article_keywords, 12)
print(top_article_keywords)

# Get all unique AI keywords (for completeness)
all_ai_keywords <- raw_data %>%
  separate_rows(AI_keywords, sep = ",\\s*") %>%
  distinct(AI_keywords) %>%
  pull(AI_keywords)

# Filter the data
filtered_data <- raw_data %>%
  # Keep only rows where article_keywords is in top 12
  filter(article_keywords %in% top_article_keywords) %>%
  # Ensure there's at least one AI keyword match
  mutate(has_ai_keyword = str_detect(AI_keywords, 
                                     paste(all_ai_keywords, collapse = "|"))) %>%
  filter(has_ai_keyword) %>%
  select(-has_ai_keyword)


# Process data into edge list - FIXED VERSION
df_edges <- filtered_data %>%
  # Convert to character and handle NAs
  mutate(
    article_keywords = as.character(article_keywords),
    AI_keywords = as.character(AI_keywords)
  ) %>%
  # Replace NA with empty string
  mutate(
    article_keywords = ifelse(is.na(article_keywords), "", article_keywords),
    AI_keywords = ifelse(is.na(AI_keywords), "", AI_keywords)
  ) %>%
  # Split keywords
  mutate(
    article_keywords = str_split(article_keywords, ",\\s*"),
    AI_keywords = str_split(AI_keywords, ",\\s*")
  ) %>%
  # Create combinations
  rowwise() %>%
  mutate(pairs = list(expand.grid(
    unlist(article_keywords), 
    unlist(AI_keywords), 
    stringsAsFactors = FALSE
  ))) %>%
  unnest(pairs) %>%
  # Filter valid pairs
  filter(Var1 != "", Var2 != "") %>%
  select(ItemA = Var1, ItemB = Var2)

# Create edges with weights
edges <- df_edges %>%
  count(ItemA, ItemB, name = "weight")

# Create nodes
nodes_itemA <- data.frame(
  id = unique(edges$ItemA),
  group = "ItemA",
  stringsAsFactors = FALSE
)

nodes_itemB <- data.frame(
  id = unique(edges$ItemB),
  group = "ItemB",
  stringsAsFactors = FALSE
)

nodes <- bind_rows(nodes_itemA, nodes_itemB)

# Calculate node frequencies
node_freq <- bind_rows(
  edges %>% group_by(id = ItemA) %>% summarise(freq = sum(weight)),
  edges %>% group_by(id = ItemB) %>% summarise(freq = sum(weight))
) %>% 
  group_by(id) %>% 
  summarise(freq = sum(freq))

nodes <- nodes %>%
  left_join(node_freq, by = "id") %>%
  mutate(freq = ifelse(is.na(freq), 1, freq))

# FILTER NODES WITH FREQUENCY > 0
nodes_filtered <- nodes %>% filter(freq > 0)

# Filter edges to only include connections between filtered nodes
edges_filtered <- edges %>%
  filter(ItemA %in% nodes_filtered$id & ItemB %in% nodes_filtered$id)

# Create graph object with proper 'type' attribute
graph <- tbl_graph(
  nodes = nodes_filtered %>%
    # Ensure only valid groups exist and convert to type
    filter(group %in% c("ItemA", "ItemB")) %>%
    mutate(type = if_else(group == "ItemA", "article", "ai")),
  edges = edges_filtered %>% rename(from = ItemA, to = ItemB),
  directed = FALSE
)

# Calculate degree centrality for label sizing
graph <- graph %>%
  mutate(degree = centrality_degree())

# Create optimized plot with improved label connections and node separation
set.seed(42)
network_plot <- ggraph(graph, layout = "fr", niter = 5000, start.temp = 0.1) + 
  geom_edge_link(
    aes(edge_width = weight),
    color = "grey40",
    alpha = 0.4,
    show.legend = FALSE
  ) + 
  geom_node_point(
    aes(size = freq, fill = group),
    shape = 21,
    color = "black",
    stroke = 0.5,
    show.legend = FALSE
  ) +
  # Enhanced labels with more visible connection lines
  geom_node_label(
    aes(label = id, fill = group),
    family = "Times New Roman",
    size = 6,  # Uniform size
    color = "black",
    alpha = 0.85,
    repel = TRUE,
    point.padding = unit(1.0, "lines"),  # Increased space around nodes
    box.padding = unit(1.2, "lines"),    # Increased space between labels
    min.segment.length = 0.1,            # More connection lines
    max.overlaps = Inf,
    segment.size = 0.8,                  # Thicker connection lines
    segment.color = "grey30",             # Darker for better visibility
    segment.alpha = 0.6,                 # More opaque
    segment.curvature = 0.2,             # Slightly curved lines
    segment.angle = 20,                  # Angled connections
    segment.ncp = 3,                     # Smoother curves
    show.legend = FALSE
  ) +
  scale_fill_manual(
    values = c(ItemA = "#449464", ItemB = "#f7be8a"),
    name = "Node Type",
    labels = c(ItemA = "Article Keywords", ItemB = "AI Keywords"),
    guide = guide_legend(override.aes = list(size = 5))
  ) +
  scale_size_continuous(
    range = c(3, 15),
    guide = "none"
  ) +
  scale_edge_width_continuous(range = c(0.5, 3)) +
  theme_void() +
  theme(
    text = element_text(family = "Times New Roman"),
    legend.position = "right",
    legend.title = element_text(face = "bold", size = 12),
    legend.text = element_text(size = 11),
    plot.margin = margin(2, 2, 2, 2, "cm"),
    panel.background = element_rect(fill = "white", color = NA)
  ) +
  labs(title = plot_title) +
  expand_limits(x = c(-1, 1), y = c(-1, 1))  # More space for spread

setwd("/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/lit_review_deepmind/plots/final_new_networks")
# Save as high-quality SVG
ggsave(
  save_file_name,
  plot = network_plot,
  device = svglite,
  width = 14,  # Increased dimensions
  height = 12,
  units = "in"
)

# Display plot
print(network_plot)