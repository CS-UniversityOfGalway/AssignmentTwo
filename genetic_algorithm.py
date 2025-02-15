"""
Genetic Algorithm TSP Solver for CT421 Project 1

Some general overview comments:
- Tournament selection picks two good parents
- Crossover creates new tour from parents
- Mutation MIGHT make a small change to that new tour
- This new tour (mutated or not) joins the population

Classes:
    GeneticAlgorithm: Main class implementing the genetic algorithm solver
"""
import random
import itertools
import time
from typing import List, Tuple
import tsplib95
import matplotlib.pyplot as plt
import pandas as pd

class GeneticAlgorithm:
    """Main class implementing the genetic algorithm solver for the TSP
    """
    def __init__(self, tsp_data, pop_size=50, generations=1000,
                 mutation_rate=0.01, crossover_rate=0.8):
        """Initializes the genetic algorithm with the provided parameters

        Args:
            tsp_data (tsplib95): The TSP data to solve
            pop_size (int, optional): Population size. Defaults to 50.
            generations (int, optional): Number of generations. Defaults to 1000.
            mutation_rate (float, optional): Chance of mutation. Defaults to 0.01.
            elite_size (int, optional): Number of elite solutions. Defaults to 2.
            crossover_rate (float, optional): Probability of crossover occurring. Defaults to 0.8.
        """
        self.tsp = tsp_data
        self.population_size = pop_size
        self.num_generations = generations
        self.mut_rate = mutation_rate
        self.elite = 2 # Number of elite solutions to keep
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_fitness_history = []
        # We are precomputing the distance matriox for faster fitness calculation later
        self.distance_matrix = self._precompute_distances()

    def _precompute_distances(self):
        """Precompute distances between all cities
        
        Returns:
            List[List[int]]: A 2D list of distances between all cities
        """
        dim = self.tsp.dimension # Number of cities in problem
        # Create a 2D matrix of zeros to store distances, with dimensions equal to number of cities
        matrix = [[0 for _ in range(dim)] for _ in range(dim)]
        # Iterting through all pairs of cities
        for i in range(1, dim + 1):
            for j in range(1, dim + 1):
                #No need to calculate distance between same city
                if i != j:
                    # Uusing TSPLIB95's get_weight method to get the distance between two cities
                    # Taking one away from the indexes for basing the indexes from 0
                    matrix[i-1][j-1] = self.tsp.get_weight(i, j)
        return matrix


    def create_initial_population(self):
        """Creates the initial population of potential solutions that the GA will evolve over time
        """
        num_cities = self.tsp.dimension # Parses the number of cities from the TSP data
        for _ in range(self.population_size): # Creates a population of random tours, defined by
                                              # the provided population size
            indiv = list(range(num_cities))# Each city is represented by a unique integer
            random.shuffle(indiv) # Randomly shuffles the order of the cities to create a
                                  # random tour
            self.population.append(indiv) # Adds the tour to the population list


    def calculate_tour_length(self, tour: list) -> float:
        """Calculate the total length of the tour.
            
        Args:
            tour (list): A list of city indices, in order that the cities are visited in the tour

        Returns:
            float: Total tour length, summed together city to city
        """
        total = 0
        # Iterating through all cities pairs in the tour
        for i in range(len(tour) - 1):
            # add distances from on to the next city and add to totalk
            total += self.distance_matrix[tour[i]][tour[i + 1]]
        # Add the distance from the last city back to the first city
        total += self.distance_matrix[tour[-1]][tour[0]]
        return total


    def calculate_fitness(self, individual: List[int]) -> float:
        """Evalutes how 'good' each soluton is designed to covert our problem from minimization
        to maximisation length. We want to minimize tour length used, also to decide which solutions
        that survive this generation, which become parents ands whats our best solution so far.

        Returns:
            float: Fitness score of the tour. Higher values indicate better solutions
              - Returns 1/tour_length to convert minimization to maximization problem
              - Returns infinity if tour length is zero (invalid tour)
        """
        try:
            # Coverting distance to fitness value
            return 1 / self.calculate_tour_length(individual) # Minimization -> maximization
        except ZeroDivisionError: # If this exception is thrown, the tour is considered invalid and
                                  # an infinite fitness is returned (Worst possible fitness)
            return float('inf')

    def get_best_individual(self) -> Tuple[List[int], float]:
        """Returns the best tour found in the current population along with its distance.

        Returns:
            Tuple[List[int], float]: A tuple containing:
            - The best tour as a list of city indices
            - total distance of this tour
        """
        # Returns the best tour
        best = max(self.population, key=self.calculate_fitness)
        # The list tour is returned, fitness is calculated and divided itno 1 to get the distance
        return best, 1 / self.calculate_fitness(best)


    def tournament_selection(self, tournament_size: int = 3) -> List[int]:
        """Essentially just picks x amount of random tours(solutions) and returns the best one.
        Randomly picks 3 tours(complete solutions) which exist in the population if we have 5
        complete solutions, this picks 3 and compares their fitness, returns the one with the
        highest fitness. This returned tour will be used as a parent to create new solution
        with cross over from another parent. Therefore the overall goal of the tournament
        selection is to pick parents for breeding.

        Args:
            tournament_size (int, optional): How many random competing solutions are chosen.
                                             Defaults to 3.

        Returns:
            List[int]: The tour (solution) with the best fitness value among the tournament_size
            randomly selected candidates.
        """
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=self.calculate_fitness)


    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Order crossover is called after tournament selection returns two parents with
        suitable fitness. It preseves chunks of consecutive cities which may be 'good-routes'
        gets size of tour so that we can pick two random points to crossover and defines
        the segment we will copy from parent1 to child

        Args:
            parent1 (List[int]): First parent tour represented as a list of city indices.
                                 Contributesa segment of its own tour to the child tour.
            parent2 (List[int]): Second parent tour represented as a list of city indices.
                                 Fills the remaining spots in the child tour. While ensuring
                                 that no city is repeated.

        Returns:
            List[int]: Complete child tour with a segment from parent1 and the rest from parent2
                       that does not contain any repeated cities.
        """
        size = len(parent1) # Get the length of parent1's tour
        start, end = sorted(random.sample(range(size), 2)) # Pick two random points to crossover
        child = [-1] * size # Creates empty child tour filled with -1
        child[start:end] = parent1[start:end] # Copies segments from parent1 into new child tour
                                              # using the random segment genetated
        # Creates an array which has ciiies that are not in the child tour
        remaining_cities = [x for x in parent2 if x not in set(parent1[start:end])]
        # for loop to fill the remaining spots in the child tour with new cities from parent2
        j = 0
        for i in range(size):
            if child[i] == -1: # If found empty spot in child
                child[i] = remaining_cities[j] # Fill empty spot with next remaining city
                j += 1
        return child


    def edge_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        """Edge crossover is a variation of order crossover that preserves edges
           between cities from the parent tours. It builds a new tour by considering
           the neighbors of each city in both parents, uses existing connections
           when possible.

        Args:
            parent1 (List[int]): First parent tour represented as a list of city indices.
                                 Used to identify neighbor relationships between cities.
            parent2 (List[int]): Second parent tour represented as a list of city indices.
                                 Used to identify neighbor relationships between cities.

        Returns:
            List[int]: A new child tour that attempts to preserve edge relationships from
            both parent1 and parent2. Starting from a random city, it builds the tour by 
            choosing next cities based on their adjacency in the parent tours. When no
            adjacent cities are available, it selects a random unused city.
        """
        size = len(parent1) # Get the length of parent1's tour
        child = [-1] * size # Creates empty child tour filled with -1

        # Start with random city from parten 1 as first city in child
        current = random.choice(parent1)
        child[0] = current

        # fill the rest of the child tour
        for i in range(1, size):
            # Parse the index of the current city in child in both parents
            p1_idx = parent1.index(current)
            p2_idx = parent2.index(current)

            # Next cities are the ones that come after the current city in both parents.
            # 1 is added to the parsed indexes from the parents.
            # Modulo is used to go back to the end of the tour if the current index exceeds the size
            next_p1 = parent1[(p1_idx + 1) % size]
            next_p2 = parent2[(p2_idx + 1) % size]

            # If both next cities are not in child, random one is choen as next city
            if next_p1 not in child and next_p2 not in child:
                current = random.choice([next_p1, next_p2])
            # If one of the parent cities are in child, the other one is chosen as next city
            # and vice versa.
            elif next_p1 not in child:
                current = next_p1
            elif next_p2 not in child:
                current = next_p2
            else:
                # If both are in child, all unused cities are parsed only from parent1 since they
                # contain the same cities in different orders
                unused = [x for x in parent1 if x not in child]
                current = random.choice(unused) # random city from list of unused cities is chosen

            child[i] = current

        return child


    def swap_mutation(self, individual: List[int]) -> List[int]:
        """Happens after crossover creates a new child. Gives small random changes to potentially
        introduce new good solutions. Provided chance of swapping two numbers after creation of a 
        new child from 2 parents to create mutation.

        Args:
            individual (List[int]): A tour represented as a list of city indices, child tour from
                                    crossover of two parents

        Returns:
            List[int]: Either the original tour if no mutation occurs, or a new tour with two cities
            swapped if it did occur.
        """
        # If random value is less than mutation rate, mutation occurs
        if random.random() < self.mut_rate:
            # Pick 2 random positions in the tour and swap the cities at those positions
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual


    def inversion_mutation(self, individual: List[int]) -> List[int]:
        """Variation of mutation that inverts a subsection of the tour if random chance hits
        Example: [1,2,3,4,5] might become [1,4,3,2,5]

        Returns:
            List[int]: Either the original tour if no mutation occurs, or a new tour with two
            subsections swapped.
        """
        # If random value is less than mutation rate, mutation occurs
        if random.random() < self.mut_rate:
            # Pick 2 random positoins
            pos1, pos2 = sorted(random.sample(range(len(individual)), 2))
            # Reverse the subsection between positions
            individual[pos1:pos2] = individual[pos1:pos2][::-1]
        return individual


    def plot_performance(self):
        """Plots the performance metrics of the genetic algorithm
        Shows both best and average fitness over generations
        """
        plt.figure(figsize=(10, 6))
        # Sequence of from 0 to number of generations that have the best fitness
        generations = range(len(self.best_fitness_history))

        # Plot best fitness
        plt.plot(generations, self.best_fitness_history, 'b-', label='Best Fitness')

        plt.xlabel('Generation')
        plt.ylabel('Fitness')
        plt.title(f'GA Performance on {self.tsp.name}')
        plt.legend()
        plt.grid(True)

        plt.show()


    def evolve(self):
        """Runs the genetic algorithm to find an optimal TSP tour. The algorithm:
            - Creates an initial random population
            - For each generation:
                - Preserves elite (best) solutions
                - Fills rest of population through:
                    Tournament selection of parents
                    Crossover 50/50 order and edge crossover
                    Chance of mutation
            - Tracks and reports progress
            
            Returns:
                Tuple[List[int], float]: A tuple containing:
                    - The best tour found across all generations
                    - The total distance of this best tour
            """
        start_time = time.time() # Note the start time of the algorithm
        self.create_initial_population() # Create the initial random population

        # Run through the amount of generations specified
        for gen in range(self.num_generations):
            # Start with new empty pop
            new_population = []

            # Keep elite individuals,
            # Sort current population by fitness and keep best individuals
            sorted_pop = sorted(self.population, key=self.calculate_fitness, reverse=True)
            new_population.extend(sorted_pop[:self.elite]) # Preserve amount of elite solutions
                                                           # specified

            # Create rest of new population
            # Then until we fill the population, pick two parents with tournament
            # Create child through crossover
            # Mutate child(small chance of random change)
            # Add to new population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()

                # First check if crossover should occur based on crossover rate
                if random.random() < self.crossover_rate:
                    # Then decide which type of crossover to use
                    if random.random() < 0.5:
                        child = self.order_crossover(parent1, parent2)
                    else:
                        child = self.edge_crossover(parent1, parent2)
                else:
                    # No crossover, just copy one parent
                    child = parent1.copy()

                # Apply mutation with random choice between operators
                if random.random() < self.mut_rate:
                    # Randomly choose between swap and inversion mutation
                    if random.random() < 0.5:
                        child = self.swap_mutation(child)
                    else:
                        child = self.inversion_mutation(child)
                new_population.append(child)

            # Replace old population with new one.
            self.population = new_population

            # Record statistics
            # Keep track of our best tour and average performance
            _,current_gen_best_length = self.get_best_individual()
            # Append the best fitness of the current generation to the history
            self.best_fitness_history.append(current_gen_best_length)

            # Print progress
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {current_gen_best_length:.2f}")

        # End time noed
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time:.2f} seconds") # Time taken
        return self.get_best_individual() # return best tour and its length


def grid_search(tsp_instance,
                population_sizes: List[int],
                crossover_chance: List[float],
                mutation_chance: List[float],
                generations: int) -> pd.DataFrame:
    """
    Performs grid search over GA parameters and returns results as DataFrame.
    
    Args:
        tsp_instance: TSP problem instance
        pop_sizes: List of population sizes to test
        crossover_rates: List of crossover rates to test
        mutation_rates: List of mutation rates to test
        generations: Number of generations to run each test
    
    Returns:
        Tuple containing:
        - DataFrame with all run results
        - Dict with best parameters found
        - Float of best tour length
        - Float of runtime for best tour
        - Float of total grid search runtime
    """
    grid_search_results = [] # List to store results, will be converted to DataFrame later
    total_start_time = time.time()
    curr_best_tour_length = float('inf')
    curr_best_run_time = None
    best_params = None
    best_fitness_history = None  # Track best fitness history

    # Generate all parameter combinations
    param_combinations = list(itertools.product(
        population_sizes,
        crossover_chance,
        mutation_chance
    ))

    total_runs = len(param_combinations)
    current_run = 0

    # Load the TSP instance once before the loop
    tsp_data = tsplib95.load(tsp_instance)

    #Loop through all parameter combinations
    for population_size, c_rate, m_rate in param_combinations:
        current_run += 1
        print(f"\nRun {current_run}/{total_runs}")
        print(f"Testing: pop={population_size}, cross={c_rate}, mut={m_rate}")

        # Create and run a GA with current parameters
        ga = GeneticAlgorithm(
            tsp_data,
            pop_size=population_size,
            generations=generations,
            mutation_rate=m_rate,
            crossover_rate=c_rate
        )

        start_time = time.time()
        _, tour_length = ga.evolve()
        run_time = time.time() - start_time

        # Update best tour length and parameters if needed
        # (i.e if the current run produced a better tour)
        if tour_length < curr_best_tour_length:
            curr_best_tour_length = tour_length
            curr_best_run_time = run_time
            best_params = {
                'population_size': population_size,
                'crossover_rate': c_rate,
                'mutation_rate': m_rate
            }
            best_fitness_history = ga.best_fitness_history.copy()# Save history of best run

        # Record results
        grid_search_results.append({
            'population_size': population_size,
            'crossover_rate': c_rate,
            'mutation_rate': m_rate,
            'run_number': current_run,
            'best_tour_length': curr_best_tour_length,
            'computation_time': run_time
        })

    # Convert results to DataFrame
    results_df = pd.DataFrame(grid_search_results)
    curr_total_runtime = time.time() - total_start_time

    best_ga = GeneticAlgorithm(tsp_data) # Create GA object for plotting
    # Set the best run history to plot
    best_ga.best_fitness_history = best_fitness_history
    best_ga.plot_performance()

    return results_df, best_params, curr_best_tour_length, curr_best_run_time, curr_total_runtime


if __name__ == "__main__":
    # --- User-defined parameters ---
    # Path to the TSP dataset folder
    DATASET_PATH = "tsp_datasets/"

    # !!Modify this to test different TSP instances!!
    PROBLEM_FILE = "berlin52.tsp"

    # !!Modify this to test generation amounts!!
    GENS = 1000

    # !!Define parameter ranges to test (MODIFY AS NEEDED)!!
    pop_sizes = [50, 100, 200]
    crossover_rates = [0.7, 0.8, 0.9]
    mutation_rates = [0.01, 0.02, 0.05]
    #----------------------------------

    print(f"\nSolving {PROBLEM_FILE}")

    # Run grid search
    results, best_config, best_tour_length, best_run_time, total_runtime = grid_search(
        DATASET_PATH + PROBLEM_FILE,
        population_sizes=pop_sizes,
        crossover_chance=crossover_rates,
        mutation_chance=mutation_rates,
        generations = GENS
    )

    # Save results to a csv file
    results.to_csv('grid_search_results.csv', index=False)

    # Print best configuration
    print("\nBest configuration found:")
    print(f"Population Size: {best_config['population_size']}")
    print(f"Crossover Rate: {best_config['crossover_rate']:.2f}")
    print(f"Mutation Rate: {best_config['mutation_rate']:.3f}")
    print(f"Best Tour Length: {best_tour_length:.2f}")
    print(f"Best Fitness Run Time: {best_run_time:.2f}s")
    print(f"\nTotal Grid Search Runtime: {total_runtime:.2f}s")
