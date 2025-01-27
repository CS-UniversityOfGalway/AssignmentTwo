import numpy as np
import tsplib95
from typing import Dict, Tuple
import os

# TSP file contains coordinates of cities(x,y positions)

# a 'tour' is a list of cities that we have to visit
# so [0,3,6,9] means visit city 0, then city 3, then city 6, then city 9 again


# distance[i][j] = distance from city i to city j
# distance[i][i] = 0
# distance[i][j] = distance[j][i]

class TSPDataLoader:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TSP file not found: {file_path}")
            
        try:
            self.problem = tsplib95.load(file_path)
        except Exception as e:
            raise ValueError(f"Error loading TSP file: {str(e)}")
        
        self.dimension = self.problem.dimension
        # This is not a function, it is a variable which holds pre calclated distances
        self.distances = self._create_distance_matrix()
        self.name = self.problem.name
    
    def _create_distance_matrix(self) -> np.ndarray:
        # _create_distance matrix returns empty distance array(of all zeroes)
        matrix = np.zeros((self.dimension, self.dimension))
        #N*N matrix of all possible city to city distances
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    try:
                        # tsplib95 uses 1-based indexing
                        matrix[i][j] = self.problem.get_weight(i + 1, j + 1)
                    except Exception as e:
                        print(f"Warning: Error getting distance for cities {i+1} and {j+1}")
                        matrix[i][j] = float('inf')
        
        return matrix
    
    def get_coordinates(self) -> Dict[int, Tuple[float, float]]:
        if not hasattr(self.problem, 'node_coords'):
            raise AttributeError("This TSP instance doesn't have coordinate data")
        # converts from 1 based indexing o 0 based indexing
        return {k-1: v for k, v in self.problem.node_coords.items()}
    
    def get_optimal_tour_length(self) -> float:
        if hasattr(self.problem, 'optimal_value'):
            return float(self.problem.optimal_value)
        return None
    
    def calculate_tour_length(self, tour: list) -> float:
        if len(tour) != self.dimension:
            raise ValueError("Tour length doesn't match problem dimfension")
            
        total = 0
        # Loop through each consecutive pair of cites
        # if aray is p0,5,3,8,1] then this loop will
        # get distance on first iteration between 0 and 5
        # Second iterations between 5 and 3
        # Third iteration between 3 and 8
        # Fourth iteration between 8 and 1
        # and so on
        for i in range(len(tour)-1):
            # tour i is the current city tour [i+1] is the next city
            total += self.distances[tour[i]][tour[i+1]]
        # Add distance back from last city to the first city, to complete loop
        # imagine we have 5->3->8->1->5
        # this line adds distance from 5 to 5, to complete the loop
        total += self.distances[tour[-1]][tour[0]]
        return total

if __name__ == "__main__":
    try:
        loader = TSPDataLoader("berlin52.tsp")
        print(f"Successfully loaded {loader.name}")
        print(f"Problem dimension: {loader.dimension}")
        print(f"Distance matrix shape: {loader.distances.shape}")
        
        #Test a sample route
        sample_route = list(range(loader.dimension))
        length = loader.calculate_tour_length(sample_route)
        print(f"Sample route length: {length}")
        
    except Exception as e:
        print(f"Error: {str(e)}")