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
        """Initializes the genetic algorithm with the provided parameters"""
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
    
    # Add this to organize your fixed strategies
        self.fixed_strategies = [
        self.always_cooperate,
        self.always_defect,
        self.tit_for_tat,
        self.suspicious_tit_for_tat
    ]

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




    def bit_flip_mutation(self, genome):
        """Bit-flip mutation for binary genomes
    
        Args:
         genome (List[int]): Genome to mutate
        
        Returns:
            List[int]: Mutated genome
        """
        mutated = genome.copy()
        for i in range(len(mutated)):
            if random.random() < self.mut_rate:
                mutated[i] = 1 - mutated[i]  # Flip bit (0->1, 1->0)
        return mutated

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






    def plot_performance(self):
        """Plot the performance of the GA over generations"""
        plt.figure(figsize=(10, 6))
        generations = range(len(self.best_fitness_history))
    
        plt.plot(generations, self.best_fitness_history, 'b-', label='Best Fitness')
        plt.plot(generations, self.avg_fitness_history, 'g-', label='Average Fitness')
    
        plt.xlabel('Generation')
        plt.ylabel('Fitness (Average Score)')
        plt.title('GA Performance on IPD Strategy Evolution')
        plt.legend()
        plt.grid(True)
        plt.savefig('ipd_evolution_performance.png')
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
    

    def analyze_strategy(self, genome):
        """Analyze a strategy and return a human-readable description
    
        Args:
            genome (List[int]): Strategy genome to analyze
        
        Returns:
            str: Human-readable description of the strategy
        """
        strategy_description = "Strategy: "
        strategy_description += "First move: " + ("C" if genome[0] == 1 else "D")
    
        if self.memory_length == 1:
            strategy_description += ", After C: " + ("C" if genome[1] == 1 else "D")
            strategy_description += ", After D: " + ("C" if genome[2] == 1 else "D")
        elif self.memory_length == 2:
            strategy_description += ", After CC: " + ("C" if genome[1] == 1 else "D")
            strategy_description += ", After CD: " + ("C" if genome[2] == 1 else "D")
            strategy_description += ", After DC: " + ("C" if genome[3] == 1 else "D")
            strategy_description += ", After DD: " + ("C" if genome[4] == 1 else "D")
    
        # Test strategy against each fixed strategy
        for strategy in self.fixed_strategies:
            strategy_name = strategy.__name__
            my_score, opp_score = self.play_game(genome, strategy)
            strategy_description += f"\nVs {strategy_name}: Score = {my_score}, Opponent = {opp_score}"
    
        return strategy_description

    def play_game(self, genome, opponent_strategy, rounds=200):
        """Play a game of Iterated Prisoner's Dilemma
    
        Args:
            genome (List[int]): Strategy genome to evaluate
            opponent_strategy: Function that takes histories and returns a move
            rounds (int): Number of rounds to play
        
        Returns:
            Tuple[int, int]: (my_score, opponent_score)
        """
        my_history = []
        opponent_history = []
        my_score = 0
        opponent_score = 0
    
        for _ in range(rounds):
            # Get moves
            my_move = self.interpret_strategy(genome, opponent_history, my_history)
            opponent_move = opponent_strategy(my_history, opponent_history)
        
            # Update histories
            my_history.append(my_move)
            opponent_history.append(opponent_move)
        
            # Update scores based on payoff matrix
            payoff = self.payoff_matrix[(my_move, opponent_move)]
            my_score += payoff[0]
            opponent_score += payoff[1]
        
        return my_score, opponent_score



if __name__ == "__main__":
    # Run with Memory-1 strategies
    print("\nEvolving Memory-1 Strategies:")
    ga_mem1 = IPDGeneticAlgorithm(
        pop_size=100,
        generations=50,
        mutation_rate=0.05,
        crossover_rate=0.8,
        memory_length=1
    )
    best_genome_mem1, best_fitness_mem1 = ga_mem1.evolve()
    ga_mem1.plot_performance()
    
    # Run with Memory-2 strategies
    print("\nEvolving Memory-2 Strategies:")
    ga_mem2 = IPDGeneticAlgorithm(
        pop_size=100,
        generations=50,
        mutation_rate=0.05,
        crossover_rate=0.8,
        memory_length=2
    )
    best_genome_mem2, best_fitness_mem2 = ga_mem2.evolve()
    ga_mem2.plot_performance()
