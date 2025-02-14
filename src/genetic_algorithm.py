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
import os
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
from tsp_loader import TSPDataLoader

class GeneticAlgorithm:
    """Main class implementing the genetic algorithm solver for the TSP
    """
    def __init__(self, tsp_instance, pop_size=50, generations=1000,
                 mutation_rate=0.01, elite_size=1, crossover_rate=0.8):
        """Initializes the genetic algorithm with the provided parameters

        Args:
            tsp_instance (TSPDataLoader): The TSP data to solve
            pop_size (int, optional): Population size. Defaults to 50.
            generations (int, optional): Number of generations. Defaults to 1000.
            mutation_rate (float, optional): Chance of mutation. Defaults to 0.01.
            elite_size (int, optional): Number of elite solutions. Defaults to 2.
            crossover_rate (float, optional): Probability of crossover occurring. Defaults to 0.8.
        """
        self.tsp = tsp_instance
        self.population_size = pop_size
        self.num_generations = generations
        self.mut_rate = mutation_rate
        self.elite = elite_size
        self.crossover_rate = crossover_rate
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []


    def create_initial_population(self):
        """Creates the initial population of potential solutions that the GA will evolve over time
        """
        num_cities = self.tsp.dimension # Parses the number of cities from the TSP data
        for _ in range(self.population_size): # Creates a population of random tours, defined by
                                              # the provided population size
            indiv = list(range(num_cities)) # Each city is represented by a unique integer
            random.shuffle(indiv) # Randomly shuffles the order of the cities to create a
                                  # random tour
            self.population.append(indiv) # Adds the tour to the population list


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
            return 1 / self.tsp.calculate_tour_length(individual) # Minimization -> maximization
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
        remaining_cities = [x for x in parent2 if x not in child[start:end]]
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
        plt.plot(generations, self.best_fitness_history, 'b-', label='Best Tour Length')

        # Plot average fitness
        plt.plot(generations, self.avg_fitness_history, 'r--', label='Average Tour Length')

        plt.xlabel('Generation')
        plt.ylabel('Tour Length')
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

                # Small chance of mutation
                child = self.swap_mutation(child)
                new_population.append(child)

            # Replace old population with new one.
            self.population = new_population

            # Record statistics
            # Keep track of our best tour and average performance
            _,current_gen_best_length = self.get_best_individual()
            # Append the best fitness of the current generation to the history
            self.best_fitness_history.append(current_gen_best_length)

            # Get all fitnesses in the population and find average
            fitnesses = [1/self.calculate_fitness(ind) for ind in self.population]
            self.avg_fitness_history.append(sum(fitnesses)/len(fitnesses))

            # Print progress
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {current_gen_best_length:.2f}")

        # End time noed
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time:.2f} seconds") # Time taken
        return self.get_best_individual() # return best tour and its length


if __name__ == "__main__":
    #TEMP: For testing purposes, only run on berlin52 for now, expand to other datasets later
    problem_files = ["berlin52.tsp"]
    # Run through each problem file
    for problem in problem_files:
        try:
            print(f"\nSolving {problem}")
            DATASET_PATH = "tsp_datasets/"
            # Check if the dataset is in the default location, if not, try the parent directory
            if not os.path.exists(DATASET_PATH + problem):
                DATASET_PATH = "../tsp_datasets/"
            # Parse tsp data from file
            tsp_data = TSPDataLoader(DATASET_PATH + problem)
            # Create an instance of the genetic algorithm chosen with the parameters
            ga = GeneticAlgorithm(tsp_data, pop_size=500, generations=1000, mutation_rate=0.01,
                                  crossover_rate=1)
            # Best route and length recorded
            best_route, best_length = ga.evolve()

            print(f"Final Fitness: {best_length:.2f}")
            print(f"Final Path: {best_route}")

            # Plot results
            ga.plot_performance()

        except FileNotFoundError:
            print(f"Could not find dataset file for {problem}")
            continue
        except ValueError as e:
            print(f"Invalid data format in {problem}: {str(e)}")
            continue
