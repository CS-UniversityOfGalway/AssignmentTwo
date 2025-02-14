"""
Genetic Algorithm TSP Solver for CT421 Project 1

Some general overview comments:
- Tournament selection picks two good parents
- Crossover creates new tour from parents
- Mutation MIGHT make a small change to that new tour
- This new tour (mutated or not) joins the population

Classes:
    GeneticAlgorithm: Main class implementing the genetic algorithm solver

Functions:
    __init__: Initializes the genetic algorithm with the provided parameters
    create_initial_population: Creates the initial population of potential solutions that the GA 
                               will evolve over time
    plot_performance: Plots the performance metrics of the genetic algorithm
    calculate_fitness: Evalutes how 'good' each soluton is designed to covert our problem from
                       minimization to maximisation length.
    tournament_selection: Essentially just picks x amount of random tours(solutions) and returns
                          the best one
    order_crossover: Order crossover is called after tournament selection returns two parents with
                     suitable fitness
    edge_crossover: Edge crossover is called after tournament selection returns two parents with
                    suitable fitness
    swap_mutation: Gives small random changes to potentially introduce new good solutions
    inversion_mutation: Inverts a subsection of the tour if random chance hits
    get_best_individual: Returns the best individual from the population
    evolve: Main function that evolves the population over generations and returns the best 
            solution
"""
import random
import os
import time
from typing import List, Tuple
import matplotlib.pyplot as plt
from EvolutionarySearch.src.tsp_loader import TSPDataLoader

class GeneticAlgorithm:
    """Main class implementing the genetic algorithm solver for the TSP
    """
    def __init__(self, tsp_instance, pop_size=50, generations=100,
                 mutation_rate=0.01, elite_size=2):
        """Initializes the genetic algorithm with the provided parameters

        Args:
            tsp_instance (TSPDataLoader): The TSP data to solve, provided in tsp_datasets folder
            pop_size (int, optional): Determines how many random solutions will be created in the
                                      initial population. Defaults to 50.
            generations (int, optional): How many times the algorithm will evolve. Defaults to 100.
            mutation_rate (float, optional): Chance of mutation. Defaults to 0.01.
            elite_size (int, optional): How many good solutions to perserve in each gen.
                                        Defaults to 2.
        """
        self.tsp = tsp_instance
        self.population_size = pop_size
        self.num_generations = generations
        self.mut_rate = mutation_rate
        self.elite = elite_size
        self.population = [] # Stores current population of solutions/tours
        self.best_fitness_history = [] # Best fitness across generations is stored
        self.avg_fitness_history = [] # Average fitness across generations is also stored

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
            return 1 / self.tsp.calculate_tour_length(individual) # Minimization -> maximization
        except ZeroDivisionError: # If this exception is thrown, the tour is considered invalid and
                                  # an infinite fitness is returned (Worst possible fitness)
            return float('inf')
    
    def tournament_selection(self, tournament_size: int = 3) -> List[int]:
        #Essentially just picks x amount of random tours(solutions) and returns the best one
        # randomly picks 3 tours(complete solutions) which exist in the population
        # if we have 5 complete solutions, this picks 3 and compares their fitness
        # returns the one with the highes fitness
        # this returned tour will be used as a parent to create new solution with cross over from another parent.
        # therefore the overall goal of the tournament selection is to pick parents for breeding
        tournament = random.sample(self.population, tournament_size)
        return max(tournament, key=self.calculate_fitness)
    
    # Order crossover is called after tournament selection returns two parents with suitable fitness
    #Order crossover is good because
    # It preseves chunks of consecutive cities which may be 'good-routes'
    def order_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        # gets size of tour so that we can pick two random points to crossover
        # defines the segment we will copy from parent1 to child
        
        # Example 1
        # parent1 = [1,2,3,4,5]
        # parent2 = [5,4,1,2,3]
        # If start=1, end=3:
        # child = [-1, 2,3, -1,-1]  # Copied 2,3 from parent1
        size = len(parent1)
        start, end = sorted(random.sample(range(size), 2))
        # Creates empty child tour filled with -1 so  [-1,-1,-1,-1,-1]
        child = [-1] * size
        # Copies segments from parent 1 into new child tour
        
        #Example 2
        # parent1 = [1, 2, 3, 4, 5, 6, 7, 8]
        #start = 2, end = 5  # We'll copy positions 2,3,4

            # parent1 segment being copied:
        #    [1, 2, |3, 4, 5,| 6, 7, 8]
        # ↓  ↓  ↓
        # #child = [-1, -1, |3, 4, 5,| -1, -1, -1]
        
        # segment from parent 1 maintains its order
        # Order preservation is important because it maintains potentially good 'sub-tours'
        child[start:end] = parent1[start:end]
        
        
        
        
        # Fill remaining positions with genes from parent2
        # basically gets the remainig ctiies from parent2 that we have not used yet
        #i.e.
        # parent2 = [2, 7, 5, 8, 1, 4, 6, 3]
        # we already used 3,4,5
        # so remaining_cities = [2, 7, 8, 1, 6]
        
        # Creates an array which has ciiies that are not in the child tour
        remaining_cities = [x for x in parent2 if x not in child[start:end]]
        j = 0
        for i in range(size):
            if child[i] == -1: # If we find empty spot in child
                child[i] = remaining_cities[j] # Fill empty spot with next remaining city
                j += 1
            # FIndal child returns valid tour where each city appears exactly once
            # Part of the tour maintians parent 1s order(segment)
            # Parent2's relative order is maintained in the rest of the tour
        return child
        # Edge crossover example
        # parent1 = [1, 2, 3, 4, 5]
        # parent2 = [5, 3, 2, 1, 4]

        # 1. Start with random city (say 1)
        # child = [1, _, _, _, _]

        # 2. Look what comes after 1 in both parents:
        # In parent1: 2 comes after 1
        # In parent2: 4 comes after 1
        # Choose one randomly (say 2)

        # child = [1, 2, _, _, _]

        # 3. Continue this process, always looking at what cities
        # come next in both parents
    def edge_crossover(self, parent1: List[int], parent2: List[int]) -> List[int]:
        size = len(parent1)
        child = [-1] * size
        
        # Start with random city from either parent
        current = random.choice(parent1)
        child[0] = current
        
        # Fill rest of tour
        for i in range(1, size):
            # Find neighbors in both parents
            p1_idx = parent1.index(current)
            p2_idx = parent2.index(current)
            
            # Get next cities in both parents
            next_p1 = parent1[(p1_idx + 1) % size]
            next_p2 = parent2[(p2_idx + 1) % size]
            
            # Choose next city that hasn't been used
            if next_p1 not in child and next_p2 not in child:
                current = random.choice([next_p1, next_p2])
            elif next_p1 not in child:
                current = next_p1
            elif next_p2 not in child:
                current = next_p2
            else:
                # If both used, pick random unused city
                unused = [x for x in parent1 if x not in child]
                current = random.choice(unused)
            
            child[i] = current
            
        return child
    
    # This happens after crossover creates a new chil
    # Gives small random changes to potentially introduce new good solutions
    def swap_mutation(self, individual: List[int]) -> List[int]:
        # Basically this has a very small chance of swapping two numbers after creation of a new child from 2 parents
        # We keep the chance small so we dont destroy good solutions too much
        if random.random() < self.mut_rate:
            idx1, idx2 = random.sample(range(len(individual)), 2)
            individual[idx1], individual[idx2] = individual[idx2], individual[idx1]
        return individual
    
    
    def inversion_mutation(self, individual: List[int]) -> List[int]:
        """
        Inverts a subsection of the tour if random chance hits
        example: [1,2,3,4,5] might become [1,4,3,2,5]
        """
        if random.random() < self.mut_rate:
            # Pick 2 random positoins
            pos1, pos2 = sorted(random.sample(range(len(individual)), 2))
            # Reverse the subsection between positions
            individual[pos1:pos2] = individual[pos1:pos2][::-1]
        return individual
    
    def get_best_individual(self) -> Tuple[List[int], float]:
        best = max(self.population, key=self.calculate_fitness)
        return best, 1 / self.calculate_fitness(best)
    
    def evolve(self):
        start_time = time.time()
        self.create_initial_population()
        
        # For each generation
        for gen in range(self.num_generations):
            # Start empty population
            new_population = []
            
            # Keep elite individuals,
            # Sort current population by fitness and keep best individuals
            # Keep best solutions
            sorted_pop = sorted(self.population, 
                              key=self.calculate_fitness, reverse=True)
            new_population.extend(sorted_pop[:self.elite])
            
            # Create rest of new population
            # Then until we fill the population, pick two parents with tournament
            # create child through crossover
            # mutate child(small chance of random change)   
            # Add to new population
            while len(new_population) < self.population_size:
                parent1 = self.tournament_selection()
                parent2 = self.tournament_selection()
                
                # 50% chance of using order crossover, 50% chance of using PMX crossover
                if random.random() < 0.5:  
                    child = self.order_crossover(parent1, parent2)
                else:
                    child = self.edge_crossover(parent1, parent2)
                child = self.swap_mutation(child)
                new_population.append(child)
            
            # Replace old population with new one.
            self.population = new_population
            
            # Record statistics
            # Keep track of our best tour and average performance
            current_gen_best_length = self.get_best_individual()
            self.best_fitness_history.append(current_gen_best_length)
            
            fitnesses = [1/self.calculate_fitness(ind) for ind in self.population]
            self.avg_fitness_history.append(sum(fitnesses)/len(fitnesses))
            
            # Print progress
            if gen % 10 == 0:
                print(f"Generation {gen}: Best Fitness = {current_gen_best_length:.2f}")
        
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        return self.get_best_individual()


if __name__ == "__main__":
    #TEMP: For testing purposes, only run on berlin52 for now, expand to other datasets later
    problem_files = ["berlin52.tsp"]
    for problem in problem_files:
        try:
            print(f"\nSolving {problem}")
            dataset_path = "tsp_datasets/"
            if not os.path.exists(dataset_path + problem):
                dataset_path = "../tsp_datasets/"
            tsp_data = TSPDataLoader(dataset_path + problem)
            ga = GeneticAlgorithm(tsp_data, pop_size=50, generations=5000)
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