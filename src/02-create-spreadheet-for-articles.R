library(wordcloud)
#library(tdm)
library(stringr)
library(dplyr)
library(textstem)
library(tm)
library(SnowballC)  
library(ggplot2)
library(stopwords)
library(jsonlite)  # For JSON export
library(sf)        # For GML export
library(bibliometrix)
library(openalexR)
library(ggplot2)  
library(tidyr)
library(RefManageR)  # For writing RIS files  
library(jsonlite)  

setwd("~/SRC/AI/Deep_mind_roundtable/Lit search/scripts")

#first do for predictions
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

combined_df_predictions_for_final_set <- combined_df %>%  
  select(doi, title, extracted_labels)

combined_df_doi <- combined_df %>%  
  select(doi)
# Extract the Last Element After Splitting the 'ID' Column  
#fix_abstracts_ids <- sapply(strsplit(combined_df$key, "/"), tail, n = 1)  
#combined_df_doi <- head(combined_df_doi, 100)
fix_abstracts_ids <- combined_df_doi
real_dois <- eval(parse(text = fix_abstracts_ids))   # Now this is a proper character vector!  
fix_abstracts_dois <- gsub("^https?://doi.org/", "", real_dois)  


# Initialize an empty list to store results  
all_abstracts <- list()  
# Define the chunk size  
chunk_size <- 30  

# Loop through in chunks  
for (i in seq(1, length(fix_abstracts_dois), by = chunk_size)) {  
  chunk_dois <- fix_abstracts_dois[i:min(i + chunk_size - 1, length(fix_abstracts_dois))]  
  
  abstracts_from_openalex_dois <- oa_fetch(  
    entity = "works",  
    doi = chunk_dois,   # <--- just the vector!  
    verbose = TRUE  
  )  
  
  if (!is.null(abstracts_from_openalex_dois) && nrow(abstracts_from_openalex_dois) > 0) {  
    all_abstracts[[length(all_abstracts) + 1]] <- abstracts_from_openalex_dois  
  }  
}  

# Combine all results  
if (length(all_abstracts) > 0) {  
  all_abstracts_df <- do.call(rbind, all_abstracts)  
  if (nrow(all_abstracts_df) > 0) {  
    write_json(all_abstracts_df, "all_abstracts.json", pretty = TRUE)  
  } else {  
    stop("No valid abstracts found to save to JSON.")  
  }  
} else {  
  stop("No abstracts were retrieved.")  
}  

# Combine all results into a single dataframe
#all_abstracts_df <- do.call(rbind, all_abstracts)
all_abstracts_df <- bind_rows(all_abstracts)
all_abstracts_df$is_manually_reviewed <- 0

all_abstracts_df$author_names <- sapply(  
  all_abstracts_df$authorships,  
  function(x) {  
    if (!is.null(x) && "display_name" %in% names(x)) {  
      paste(x$display_name, collapse = ", ")  
    } else {  
      NA_character_  
    }  
  }  
)  

all_abstracts_df$topic_names <- sapply(  
  all_abstracts_df$topics,  
  function(x) {  
    if (!is.null(x) && "display_name" %in% names(x)) {  
      paste(x$display_name, collapse = ", ")  
    } else {  
      NA_character_  
    }  
  }  
)  

all_abstracts_df$keyword_names <- sapply(  
  all_abstracts_df$keywords,  
  function(x) {  
    if (!is.null(x) && "display_name" %in% names(x)) {  
      paste(x$display_name, collapse = ", ")  
    } else {  
      NA_character_  
    }  
  }  
)

all_abstracts_df_completed <- all_abstracts_df %>%  
  select(id, doi, title, is_manually_reviewed, author_names, topic_names, keyword_names)

merged_df_predictions <- merge(  
  combined_df_predictions_for_final_set,  
  all_abstracts_df_completed,  
  by = c("doi", "title"),  
  all.x = TRUE  
)  





#now do for training set
df_global <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/!ready2go/include - split by topics/topic1_global_included.csv')
df_global$extracted_labels <- 'global'
df_earth <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/!ready2go/include - split by topics/topic2_earth_included.csv')
df_earth$extracted_labels <- 'earth'
df_ocean <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/!ready2go/include - split by topics/topic3_ocean_included.csv')
df_ocean$extracted_labels <- 'ocean'
df_water <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/!ready2go/include - split by topics/topic4_water_included.csv')
df_water$extracted_labels <- 'water'
df_biodiversity <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/!ready2go/include - split by topics/topic5_biodiversity_included.csv')
df_biodiversity$extracted_labels <- 'biodiversity'
df_urban <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/!ready2go/include - split by topics/topic8_urban_included.csv')
df_urban$extracted_labels <- 'urban'
df_comms <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/!ready2go/include - split by topics/topic6_comms_included.csv')
df_comms$extracted_labels <- 'comms'
df_policy <-read.csv('/home/erikz/SRC/AI/Deep_mind_roundtable/Lit search/training_set_split8/!ready2go/include - split by topics/topic7_policy_included.csv')
df_policy$extracted_labels <- 'policy'

combined_df_training_set <- bind_rows(df_global, df_earth, df_ocean, df_water, df_biodiversity,
                         df_urban, df_comms, df_policy)

combined_df_training_set_doi <- combined_df_training_set %>%  
  select(doi)

combined_df_training_set_for_final_set <- combined_df_training_set %>%  
  select(doi, title, extracted_labels)

#function
fetch_abstracts_by_doi <- function(combined_df_doi, chunk_size = 30, write_json_path = "all_abstracts.json") {  
  # Try to convert input to a vector of DOIs  
  if (is.data.frame(combined_df_doi)) {  
    # Try to find a column with "doi" in its name  
    doi_col <- grep("doi", names(combined_df_doi), ignore.case = TRUE, value = TRUE)  
    if (length(doi_col) == 0) stop("No DOI column found in data frame.")  
    real_dois <- as.character(combined_df_doi[[doi_col[1]]])  
  } else if (is.character(combined_df_doi) && length(combined_df_doi) == 1 && grepl("^c\\(", combined_df_doi)) {  
    # If it's a string like 'c("https://doi.org/xyz", ...)'  
    real_dois <- eval(parse(text = combined_df_doi))  
  } else if (is.character(combined_df_doi)) {  
    # It's already a character vector  
    real_dois <- combined_df_doi  
  } else {  
    stop("Input must be a data frame with a DOI column, or a character vector of DOIs, or a parseable R vector string.")  
  }  
  
  # Remove the URL prefix from DOIs if present  
  fix_abstracts_dois <- gsub("^https?://doi.org/", "", real_dois)  
  
  all_abstracts <- list()  
  
  for (i in seq(1, length(fix_abstracts_dois), by = chunk_size)) {  
    chunk_dois <- fix_abstracts_dois[i:min(i + chunk_size - 1, length(fix_abstracts_dois))]  
    
    abstracts_from_openalex_dois <- oa_fetch(  
      entity = "works",  
      doi = chunk_dois,  
      verbose = TRUE  
    )  
    
    if (!is.null(abstracts_from_openalex_dois) && nrow(abstracts_from_openalex_dois) > 0) {  
      all_abstracts[[length(all_abstracts) + 1]] <- abstracts_from_openalex_dois  
    }  
  }  
  
  # Optionally combine and write as JSON (side effect, not return value)  
  if (length(all_abstracts) > 0) {  
    all_abstracts_df <- dplyr::bind_rows(all_abstracts)  
    if (nrow(all_abstracts_df) > 0) {  
      jsonlite::write_json(all_abstracts_df, write_json_path, pretty = TRUE)  
    } else {  
      warning("No valid abstracts found to save to JSON.")  
    }  
  } else {  
    warning("No abstracts were retrieved.")  
  }  
  
  return(all_abstracts)  
}  

#call it
# If you have a data frame with a column of DOI URLs:  
all_abstracts_training_set <- fetch_abstracts_by_doi(combined_df_training_set_doi)
all_abstracts_training_set_df <- bind_rows(all_abstracts_training_set)
all_abstracts_training_set_df$is_manually_reviewed <- 1

all_abstracts_training_set_df$author_names <- sapply(  
  all_abstracts_training_set_df$authorships,  
  function(x) {  
    if (!is.null(x) && "display_name" %in% names(x)) {  
      paste(x$display_name, collapse = ", ")  
    } else {  
      NA_character_  
    }  
  }  
)  

all_abstracts_training_set_df$topic_names <- sapply(  
  all_abstracts_training_set_df$topics,  
  function(x) {  
    if (!is.null(x) && "display_name" %in% names(x)) {  
      paste(x$display_name, collapse = ", ")  
    } else {  
      NA_character_  
    }  
  }  
)  

all_abstracts_training_set_df$keyword_names <- sapply(  
  all_abstracts_training_set_df$keywords,  
  function(x) {  
    if (!is.null(x) && "display_name" %in% names(x)) {  
      paste(x$display_name, collapse = ", ")  
    } else {  
      NA_character_  
    }  
  }  
)

all_abstracts_training_set_df_completed <- all_abstracts_training_set_df %>%  
  select(id, doi, title, is_manually_reviewed, author_names, topic_names, keyword_names)

merged_df_training_set <- merge(  
  combined_df_training_set_for_final_set,  
  all_abstracts_training_set_df_completed,  
  by = c("doi", "title"),  
  all.x = TRUE  
)  

#save all
merged_df_final_set <- bind_rows(merged_df_training_set, merged_df_predictions)
colnames(merged_df_final_set) <- c('doi', 'title', 'issue_area', 'openalex_id', 'is_manually_reviewed', 'authors', 'openalex_topics', 'keywords')
# Save the data frame to a CSV file
write.csv(merged_df_final_set, file = "../results/final_set_without_AI.csv", row.names = FALSE)

#function to make it work with shiny app
library(dplyr)
library(tidyr)
library(stringr)

prepare_network_data <- function(input_df) {  
  processed <- input_df %>%  
    mutate(id = row_number()) %>%  
    filter(!is.na(AI_word)) %>%  
    filter(!is.na(keywords_in_scientific_area)) %>%  
    mutate(across(c(AI_word, keywords_in_scientific_area),   
                  ~ str_replace_all(., "\"", ""))) %>%  
    separate_rows(AI_word, sep = ",") %>%  
    mutate(AI_word = str_trim(AI_word)) %>%  
    filter(AI_word != "") %>%  
    separate_rows(keywords_in_scientific_area, sep = ",") %>%  
    mutate(keywords_in_scientific_area = str_trim(keywords_in_scientific_area)) %>%  
    filter(keywords_in_scientific_area != "") %>%  
    distinct(id, AI_word, keywords_in_scientific_area, .keep_all = TRUE) %>%  
    group_by(id, AI_word, keywords_in_scientific_area, issue_area_category) %>%  
    summarise(frequency = n(), .groups = "drop") %>%  
    select(id, AI_word, keywords_in_scientific_area, frequency, issue_area_category)  
  
  return(processed)  
}  
#topic names it needs
#AI_word	keywords_in_scientific_area	frequency	issue_area_category

# Example usage
#processed_data <- merged_df_final_set %>% 
#  prepare_network_data()

# Save for Shiny app
#readr::write_csv(processed_data, "network_data.csv")