from daacs.infrastructure.rda_loader import RDALoader

#todo -> fix this loader to be relative. 
loader = RDALoader("/Users/afraser/Documents/src/daacs-nlp/src/nogit_wgu.rda")

# List all the data frame names in the R data file
data_frame_names = loader.show_all_dataframes()
print("Data frame names:", data_frame_names)

# Load a specific data frame from the R data file
# df = loader.get_df("data_frame_name")f