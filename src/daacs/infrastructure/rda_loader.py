import pyreadr

class RDALoader:
    def __init__(self, file_path):
        """
        Initialize the RDALoader with the path to an R data file.

        Args:
            file_path (str): The path to the R data file.
        """
        self.file_path = file_path
        self.data = None  # Initialize data as None

    def show_all_dataframes(self):
        """
        List all the data frame names in the R data file.

        Returns:
            list of str: A list of data frame names.
        """
        if self.data is None:
            # Load the R data file if it hasn't been loaded already
            self.data = pyreadr.read_r(self.file_path)
        
        return list(self.data.keys())

    def get_df(self, data_frame_name):
        """
        Load a specific data frame from the R data file.

        Args:
            data_frame_name (str): The name of the data frame to load.

        Returns:
            pandas.DataFrame: The loaded data frame.
        """
        if self.data is None:
            # Load the R data file if it hasn't been loaded already
            self.data = pyreadr.read_r(self.file_path)
        
        if data_frame_name not in self.data:
            raise ValueError(f"Data frame '{data_frame_name}' not found in the R data file.")
        
        return self.data[data_frame_name]