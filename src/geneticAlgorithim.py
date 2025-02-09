import numpy as np
import random
from typing import List, Tuple
import time
import matplotlib.pyplot as plt
from tspLoader import TSPDataLoader
# Tournament selection picks two good parents
# Crossover creates new tour from parents
# Mutation MIGHT make a small change to that new tour
# This new tour (mutated or not) joins the population
class GeneticAlgorithm:
    def __init__(self, tsp_data, pop_size=50, generations=100, 
                 mutation_rate=0.01, elite_size=2):
        self.tsp = tsp_data
        self.population_size = pop_size
        self.num_generations = generations
        self.mut_rate = mutation_rate
        self.elite = elite_size
        self.population = []
        self.best_fitness_history = []
        self.avg_fitness_history = []
        
    def create_initial_population(self) -> None:
        num_cities = self.tsp.dimension
        for _ in range(self.population_size):
            # Create random permutation of cities
            indiv = list(range(num_cities))
            random.shuffle(indiv)
            self.population.append(indiv)
    
    # Evalutes how 'good' each soltuon is
    # designed tocnovert out problem form minimization to mazimisation
    # we want to minimize tour length
    # used also to decide which soltutions to survvie this generation, which become parents, whats our best solution so far.
    def calculate_fitness(self, individual: List[int]) -> float:
        try:
            return 1 / self.tsp.calculate_tour_length(individual)
        except ZeroDivisionError:
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
            best_tour, best_length = self.get_best_individual()
            self.best_fitness_history.append(best_length)
            
            fitnesses = [1/self.calculate_fitness(ind) for ind in self.population]
            self.avg_fitness_history.append(sum(fitnesses)/len(fitnesses))
            
            # Print progress
            if gen % 10 == 0:
                print(f"Generation {gen}: Best tour length = {best_length:.2f}")
        
        end_time = time.time()
        print(f"\nTime taken: {end_time - start_time:.2f} seconds")
        return self.get_best_individual()

def plot_performance(self):
    """
    Plots the performance metrics of the genetic algorithm
    Shows both best and average fitness over generations
    """
    plt.figure(figsize=(10, 6))
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
    
    # Optinal: Add optimal tour length if available
    optimal = self.tsp.get_optimal_tour_length()
    if optimal:
        plt.axhline(y=optimal, color='g', linestyle=':', label='Optimal Length')
        plt.legend()
    
    plt.show()


# Modified main section for testing
if __name__ == "__main__":

    
    # Test with different problem sizes
    problem_files = ["berlin52.tsp", "kroA100.tsp", "pr1002.tsp","test.tsp"]
    
    for problem in problem_files:
        try:
            print(f"\nSolving {problem}")
            tsp_data = TSPDataLoader(problem)
            ga = GeneticAlgorithm(tsp_data, pop_size=100, generations=500)
            best_route, best_length = ga.evolve()
            
            print(f"Best tour length: {best_length:.2f}")
            if tsp_data.get_optimal_tour_length():
                print(f"Optimal tour length: {tsp_data.get_optimal_tour_length()}")
            
            # Plot results
            ga.plot_performance()
            
        except Exception as e:
            print(f"Error processing {problem}: {str(e)}")
            continue