import numpy as np
import tsplib95
from typing import Dict, Tuple
import os

class TSPDataLoader:
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"TSP file not found: {file_path}")
            
        try:
            self.problem = tsplib95.load(file_path)
        except Exception as e:
            raise ValueError(f"Error loading TSP file: {str(e)}")
        
        self.dimension = self.problem.dimension
        self.distances = self._create_distance_matrix()
        self.name = self.problem.name
    
    def _create_distance_matrix(self) -> np.ndarray:
        matrix = np.zeros((self.dimension, self.dimension))
        for i in range(self.dimension):
            for j in range(self.dimension):
                if i != j:
                    try:
                        matrix[i][j] = self.problem.get_weight(i + 1, j + 1)
                    except Exception as e:
                        print(f"Warning: Error getting distance for cities {i+1} and {j+1}")
                        matrix[i][j] = float('inf')
        return matrix
    
    def get_coordinates(self) -> Dict[int, Tuple[float, float]]:
        if not hasattr(self.problem, 'node_coords'):
            raise AttributeError("This TSP instance doesn't have coordinate data")
        return {k-1: v for k, v in self.problem.node_coords.items()}
    
    def get_optimal_tour_length(self) -> float:
        if hasattr(self.problem, 'optimal_value'):
            return float(self.problem.optimal_value)
        return None
    
    def calculate_tour_length(self, tour: list) -> float:
        if len(tour) != self.dimension:
            raise ValueError("Tour length doesn't match problem dimension")
            
        total = 0
        for i in range(len(tour)-1):
            total += self.distances[tour[i]][tour[i+1]]
        total += self.distances[tour[-1]][tour[0]]
        return total

if __name__ == "__main__":
    try:
        loader = TSPDataLoader("test.tsp")
        print(f"Successfully loaded {loader.name}")
        print(f"Problem dimension: {loader.dimension}")
        print(f"Distance matrix shape: {loader.distances.shape}")
        
        sample_route = list(range(loader.dimension))
        length = loader.calculate_tour_length(sample_route)
        print(f"Sample route length: {length}")
        
    except Exception as e:
        print(f"Error: {str(e)}")