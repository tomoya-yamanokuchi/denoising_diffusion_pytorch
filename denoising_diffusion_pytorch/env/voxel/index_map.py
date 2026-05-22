import numpy as np


class IndexMap():
    """Maps 1D indices to 2D coordinates and vice versa.

    This class provides functionality to convert between a 1D index and its corresponding
    2D coordinates (row, column) in a grid. The grid is assumed to be square, with the
    side length specified in the grid_config dictionary.

    Attributes:
        to_2d_loc (dict): A dictionary mapping 1D indices to 2D (i, j) coordinates.
        to_1d_loc (dict): A dictionary mapping 2D (i, j) coordinates to 1D indices.
    """

    def __init__(self,grid_config):

        """
        Initializes the IndexMap class with the grid configuration.

        Args:
            grid_config (dict): A dictionary containing the configuration of the grid.
                It must contain the "side_length" key, which specifies the side length of
                the square grid. The value should be a positive integer.

        Raises:
            NotImplementedError: If the side length is not a perfect square.


        Examples:
            >>> s_grid_config = {"bounds":(-0.05,0.05,-0.05,0.05,-0.05,0.05),
                                "side_length":16}
        """

        image_length = np.sqrt(grid_config["side_length"])
        if image_length.is_integer():
            image_length= int(image_length)
        else:
            NotImplementedError()

        to_2d_loc ={}
        to_1d_loc ={}

        k = 0
        for i in range(image_length):
            for j in range(image_length):
                to_2d_loc.update({k:(i,j)})
                to_1d_loc.update({(i,j):k})
                k+=1

        self.to_2d_loc = to_2d_loc
        self.to_1d_loc = to_1d_loc

    def map_1d_to_2d_loc(self,data):

        """Maps a 1D index to a 2D coordinate.

        Args:
            data (int): A 1D index to be converted to a 2D coordinate.

        Returns:
            tuple: A 2D coordinate (i, j) corresponding to the 1D index.

        Raises:
            KeyError: If the 1D index is not found in the mapping.
        """


        return self.to_2d_loc[data]


    def map_2d_to_1d_loc(self,data):
        """Maps a 2D coordinate to a 1D index.

        Args:
            data (tuple): A tuple representing the 2D coordinate (i, j) to be converted
                to a 1D index.

        Returns:
            int: The 1D index corresponding to the 2D coordinate.

        Raises:
            KeyError: If the 2D coordinate is not found in the mapping.
        """

        return self.to_1d_loc[data]

