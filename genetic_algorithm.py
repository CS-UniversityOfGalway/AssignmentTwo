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

class IPDGeneticAlgorithm:
    """Main class implementing the genetic algorithm solver for the Iterated Prisoner's Dilemma
    """
    def __init__(self, pop_size=50, generations=100,
                 mutation_rate=0.01, crossover_rate=0.8, memory_length=1):
        """Initializes the genetic algorithm with the provided parameters

        Args:
            pop_size (int, optional): Population size. Defaults to 50.
            generations (int, optional): Number of generations. Defaults to 100.
            mutation_rate (float, optional): Chance of mutation. Defaults to 0.01.
            crossover_rate (float, optional): Probability of crossover occurring. Defaults to 0.8.
            memory_length (int, optional): Length of memory for strategies. Defaults to 1.
        """
            # Add this to organize your fixed strategies
        self.fixed_strategies = [
        self.always_cooperate,
        self.always_defect,
        self.tit_for_tat,
        self.suspicious_tit_for_tat
    ]
        self.population_size = pop_size
        self.num_generations = generations
        self.mut_rate = mutation_rate
        self.elite = 2  # Number of elite solutions to keep
        self.crossover_rate = crossover_rate
        self.memory_length = memory_length
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
        # Define payoff matrix for Prisoner's Dilemma
        self.payoff_matrix = {
            ('C', 'C'): (3, 3),  # Both cooperate
            ('C', 'D'): (0, 5),  # Player 1 cooperates, Player 2 defects
            ('D', 'C'): (5, 0),  # Player 1 defects, Player 2 cooperates
            ('D', 'D'): (1, 1)   # Both defect
        }

    # opponent history and my history passed to strategies which dont use them
    # for consitency, we would need to handle different function signatures when playing games
    def always_cooperate(self, opponent_history, my_history):
        """Strategy that always cooperates"""
        return 'C'
    
    def always_defect(self, opponent_history, my_history):
        """Strategy that always defects"""
        return 'D'
    
    def tit_for_tat(self, opponent_history, my_history):
        """Strategy that begins by cooperating and then mimics opponent's previous move"""
        if not opponent_history:  # First move
            return 'C'
        return opponent_history[-1]
    

    def suspicious_tit_for_tat(self, opponent_history, my_history):
        """Strategy that begins by defecting and then mimics opponent's previous move"""
        if not opponent_history:  # First move
            return 'D'
        return opponent_history[-1]


    def create_initial_population(self):
        """Creates the initial population of potential strategies
        
        For memory-1 strategies, the genome has 3 elements:
        - First move (0=Defect, 1=Cooperate)
        - Response to opponent's cooperation (0=Defect, 1=Cooperate)
        - Response to opponent's defection (0=Defect, 1=Cooperate)
        
        For memory-2 strategies, the genome has 5 elements:
        - First move
        - Response to opponent's CC
        - Response to opponent's CD
        - Response to opponent's DC
        - Response to opponent's DD
        """
        # Calculate genome length based on memory length
        genome_length = 1 + 2**self.memory_length
        
        # Create random initial population
        for _ in range(self.population_size):
            # Each gene is 0 (Defect) or 1 (Cooperate)
            genome = [random.randint(0, 1) for _ in range(genome_length)]
            self.population.append(genome)



    def interpret_strategy(self, genome, opponent_history, my_history):
        """Interpret a genome as a strategy and return the next move
        
        Args:
            genome (List[int]): Binary genome representing the strategy
            opponent_history (List[str]): History of opponent's moves
            my_history (List[str]): History of my moves
            
        Returns:
            str: 'C' for cooperate or 'D' for defect
        """
        # First move
        if not opponent_history:
            return 'C' if genome[0] == 1 else 'D'
        
        if self.memory_length == 1:
            # Memory-1: Just consider opponent's last move
            last_move = opponent_history[-1]
            if last_move == 'C':
                return 'C' if genome[1] == 1 else 'D'  # Response to cooperation
            else:
                return 'C' if genome[2] == 1 else 'D'  # Response to defection
                
        elif self.memory_length == 2:
            # Memory-2: Consider last two moves if available
            if len(opponent_history) == 1:
                # Only one move in history, use memory-1 part of strategy
                last_move = opponent_history[0]
                if last_move == 'C':
                    return 'C' if genome[1] == 1 else 'D'
                else:
                    return 'C' if genome[2] == 1 else 'D'
            else:
                # Use last two moves
                last_two = opponent_history[-2:]
                
                if last_two == ['C', 'C']:
                    return 'C' if genome[1] == 1 else 'D'
                elif last_two == ['C', 'D']:
                    return 'C' if genome[2] == 1 else 'D'
                elif last_two == ['D', 'C']:
                    return 'C' if genome[3] == 1 else 'D'
                else:  # ['D', 'D']
                    return 'C' if genome[4] == 1 else 'D'


    def calculate_fitness(self, genome):
        """Calculate fitness by playing against all fixed strategies
    
        Args:
            genome (List[int]): The strategy genome to evaluate
        
        Returns:
            float: Average score against all fixed strategies
        """
        total_score = 0
    
        # Play against each fixed strategy
        for strategy in self.fixed_strategies:
            my_score, _ = self.play_game(genome, strategy)
            total_score += my_score
    
        # Return average score
        return total_score / len(self.fixed_strategies)

    def get_best_individual(self):
        """Return the best strategy in the current population
    
        Returns:
            Tuple[List[int], float]: Best genome and its fitness
        """
        best_genome = max(self.population, key=self.calculate_fitness)
        best_fitness = self.calculate_fitness(best_genome)
        return best_genome, best_fitness


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
        used_cities = set(parent1[start:end])
        remaining_cities = []
        for city in parent2:
            if city not in used_cities:
                remaining_cities.append(city)

        # for loop to fill the remaining spots in the child tour with new cities from parent2
        j = 0
        for i in range(size):
            if child[i] == -1: # If found empty spot in child
                child[i] = remaining_cities[j] # Fill empty spot with next remaining city
                j += 1
        return child

    def bit_flip_mutatuion(self, genom):
        return 0

    def single_point_crossover(self, parent1, parent2):
        """Single-point crossover for binary genomes
    
        Args:
            parent1 (List[int]): First parent genome
            parent2 (List[int]): Second parent genome
        
        Returns:
            List[int]: Child genome
        """
        if random.random() > self.crossover_rate:
            return parent1.copy()
        
        crossover_point = random.randint(1, len(parent1) - 1)
        child = parent1[:crossover_point] + parent2[crossover_point:]
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
        """Run the genetic algorithm to evolve IPD strategies"""
        start_time = time.time()
        self.create_initial_population()
    
        # Track best fitness over generations
        self.best_fitness_history = []
        self.avg_fitness_history = []
    
        for gen in range(self.num_generations):
            # Calculate fitness for all individuals
            fitness_scores = [self.calculate_fitness(genome) for genome in self.population]
            avg_fitness = sum(fitness_scores) / len(fitness_scores)
            self.avg_fitness_history.append(avg_fitness)
        
            # Get the best individual of this generation
            best_genome, best_fitness = self.get_best_individual()
            self.best_fitness_history.append(best_fitness)
        
            # Create a new population
            new_population = []
        
            # Add elite individuals
            sorted_population = sorted(self.population, 
                                   key=self.calculate_fitness, 
                                   reverse=True)
            new_population.extend(sorted_population[:self.elite])
        
            # Fill the rest of the population with children
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
            
                child = self.single_point_crossover(parent1, parent2)
                child = self.bit_flip_mutation(child)
            
                new_population.append(child)
        
            self.population = new_population
        
            # Print progress
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {best_fitness:.2f}, Avg Fitness = {avg_fitness:.2f}")
    
        end_time = time.time()
        print(f"\nEvolution completed in {end_time - start_time:.2f} seconds")
    
        best_genome, best_fitness = self.get_best_individual()
        strategy_description = self.analyze_strategy(best_genome)
    
        print("\nBest Strategy Found:")
        print(strategy_description)
        print(f"Average Score: {best_fitness:.2f}")
    
        return best_genome, best_fitness


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
            'best_tour_length': tour_length,
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
    PROBLEM_FILE = "pr1002.tsp"

    # !!Modify this to change the generation limit!!
    GEN_LIMIT = 1000

    # !!Define parameter ranges to test (MODIFY AS NEEDED)!!
    pop_sizes = [200, 225, 250]
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
        generations = GEN_LIMIT
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
