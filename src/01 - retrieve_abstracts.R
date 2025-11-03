#Script to fetch abstracts from openalex based on openalex ids

your_email <- "your_email@domain.com"
your_csv <- 'path_to_yourdata.csv'
your_csv_with_abstracts <- 'path_to_yourdata_with_abstracts.csv'

# Installation of Required Packages  
#install.packages("remotes")  
#remotes::install_github("ropensci/openalexR")  
#options(openalexR.mailto = your_email)

# Load Necessary Libraries  
library(openalexR)  
library(dplyr)  
library(ggplot2)  
library(tidyr)  

# Read CSV
litsearch_df <- read.csv(your_csv)

# Define the Types to Select  
types_to_select <- c('article', 'preprint', 'book-chapter', 'review',   
                     'dissertation', 'book', 'other', 'report', 'letter')  

# Filter the Data Frame Based on the Selected Types  
litsearch_df <- litsearch_df %>%   
  filter(type %in% types_to_select)  

df_fix_abstracts <- litsearch_df %>%  
  filter(is.na(abstract) | abstract == "" | trimws(abstract) == "")

# Extract the Last Element After Splitting the 'ID' Column  
fix_abstracts_ids <- sapply(strsplit(df_fix_abstracts$id, "/"), tail, n = 1)  

# Initialize an empty list to store results  
all_abstracts <- list()  

# Define the chunk size  
chunk_size <- 30  

# Loop through the fix_abstracts_ids in chunks  
for (i in seq(1, length(fix_abstracts_ids), by = chunk_size)) {  
  # Create a chunk of IDs  
  chunk_ids <- fix_abstracts_ids[i:min(i + chunk_size - 1, length(fix_abstracts_ids))]  
  
  # Retrieve Abstract Information for the current chunk  
  abstracts_from_openalex_ids <- oa_fetch(  
    entity = "works",  
    id = chunk_ids,  
    verbose = TRUE  
  )  
  
  # Select the relevant columns  
  selected_columns <- abstracts_from_openalex_ids[c("id", "abstract")]  
  
  # Append the results to the list  
  all_abstracts[[length(all_abstracts) + 1]] <- selected_columns  
}  

# Combine all results into a single data frame  
combined_abstracts <- do.call(rbind, all_abstracts)  

# Update the abstract in litsearch_df based on combined_abstracts  
litsearch_df_new <- litsearch_df %>%  
  left_join(combined_abstracts, by = "id", suffix = c("", ".new")) %>%  
  mutate(abstract = ifelse(!is.na(abstract.new), abstract.new, abstract)) %>%  
  select(-abstract.new)  # Remove the temporary column

new_df <- litsearch_df_new %>%  
  select(id, abstract)

# Write the combined data frame to a CSV file  
write.csv(litsearch_df_new, your_csv_with_abstracts, row.names = FALSE) 
