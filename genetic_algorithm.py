"""
Genetic Algorithm IPD Solver for CT421 Project 2

Some general overview comments:
- Tournament selection picks two good strategies (parents)
- Crossover creates a new strategy from the parents
- Mutation MIGHT make a small change to that new strategy
- This new strategy (mutated or not) joins the population

Classes:
    IPDGeneticAlgorithm: Main class implementing the genetic algorithm solver for the Iterated Prisoner's Dilemma
"""
import random
import itertools
import time
from typing import List
import matplotlib.pyplot as plt
import pandas as pd

class IPDGeneticAlgorithm:
    """Main class implementing the genetic algorithm solver for the Iterated Prisoner's Dilemma
    """
    def __init__(self, pop_size=50, generations=100,
             mutation_rate=0.01, crossover_rate=0.8, memory_length=1, noise_level=0.0):
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
        self.noise_level = noise_level  # Probability of a move being misinterpreted
    
    # Define payoff matrix for Prisoner's Dilemma
        self.payoff_matrix = {
        ('C', 'C'): (3, 3),  # Both cooperate
        ('C', 'D'): (0, 5),  # Player 1 cooperates, Player 2 defects
        ('D', 'C'): (5, 0),  # Player 1 defects, Player 2 cooperates
        ('D', 'D'): (1, 1)   # Both defect
    }
    
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



    def grid_search(self):
        """
        Perform a grid search to find optimal parameters for the IPD genetic algorithm.
        Tests combinations of population size, mutation rate, and crossover rate.
        """
        
        # Define parameter ranges to search
        param_grid = {
            'pop_size': [50, 100],
            'mutation_rate': [0.01, 0.05, 0.1],
            'crossover_rate': [0.7, 0.8, 0.9],
            'memory_length': [1, 2]  # Test both memory lengths
        }
        
        # Generate all combinations of parameters
        param_combinations = list(itertools.product(
            param_grid['pop_size'],
            param_grid['mutation_rate'],
            param_grid['crossover_rate'],
            param_grid['memory_length']
        ))
        
        # Set up results tracking
        results = []
        
        # Fixed number of generations for all runs
        generations = 30  # Using fewer generations to make grid search faster
        
        print(f"Running grid search with {len(param_combinations)} parameter combinations")
        start_time = time.time()
        
        # Test each parameter combination
        for i, (pop_size, mutation_rate, crossover_rate, memory_length) in enumerate(param_combinations):
            print(f"\nTesting combination {i+1}/{len(param_combinations)}:")
            print(f"Pop size: {pop_size}, Mutation rate: {mutation_rate}, Crossover rate: {crossover_rate}, Memory length: {memory_length}")
            
            # Initialize and run the GA with these parameters
            ga = IPDGeneticAlgorithm(
                pop_size=pop_size,
                generations=generations,
                mutation_rate=mutation_rate,
                crossover_rate=crossover_rate,
                memory_length=memory_length
            )
            
            best_genome, best_fitness = ga.evolve()
            
            # Save results
            results.append({
                'pop_size': pop_size,
                'mutation_rate': mutation_rate,
                'crossover_rate': crossover_rate,
                'memory_length': memory_length,
                'best_fitness': best_fitness,
                'final_avg_fitness': ga.avg_fitness_history[-1]
            })
        
        end_time = time.time()
        print(f"Grid search completed in {end_time - start_time:.2f} seconds")
        
        # Convert results to DataFrame and sort by best fitness
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('best_fitness', ascending=False)
        
        print("\nTop 5 parameter combinations:")
        print(results_df.head(5).to_string(index=False))
        
        # Save results to CSV
        results_df.to_csv('ipd_grid_search_results.csv', index=False)
        
        # Visualize results
        self.plot_grid_search_results(results_df)
        
        return results_df

    def plot_grid_search_results(self, results_df):
        """
        Create visualizations of grid search results
        
        Args:
            results_df (pd.DataFrame): DataFrame with grid search results
        """
        import matplotlib.pyplot as plt
        
        # Plot 1: Scatter plot of mutation rate vs fitness, colored by population size
        plt.figure(figsize=(10, 6))
        for pop_size in results_df['pop_size'].unique():
            subset = results_df[results_df['pop_size'] == pop_size]
            plt.scatter(subset['mutation_rate'], subset['best_fitness'], 
                       label=f'Pop size: {pop_size}', s=50, alpha=0.7)
        
        plt.xlabel('Mutation Rate')
        plt.ylabel('Best Fitness Score')
        plt.title('Effect of Mutation Rate on Best Fitness')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('grid_search_mutation_rate.png')
        plt.close()
        
        # Plot 2: Grouped bar chart for memory length comparison
        mem_group = results_df.groupby('memory_length').agg({'best_fitness': ['mean', 'max']})
        mem_group.columns = ['Mean Fitness', 'Max Fitness']
        mem_group.plot(kind='bar', figsize=(8, 5))
        plt.xlabel('Memory Length')
        plt.ylabel('Fitness Score')
        plt.title('Impact of Memory Length on Strategy Performance')
        plt.grid(True, alpha=0.3)
        plt.savefig('grid_search_memory_length.png')
        plt.close()
        
        # Plot 3: Heatmap of crossover rate vs mutation rate (for memory_length=1)
        plt.figure(figsize=(8, 6))
        for mem_length in [1, 2]:
            subset = results_df[results_df['memory_length'] == mem_length]
            pivot = subset.pivot_table(
                index='mutation_rate',
                columns='crossover_rate',
                values='best_fitness',
                aggfunc='mean'
            )
            
            plt.figure(figsize=(8, 6))
            plt.imshow(pivot, cmap='viridis', aspect='auto', interpolation='nearest')
            plt.colorbar(label='Average Best Fitness')
            plt.xticks(range(len(pivot.columns)), pivot.columns)
            plt.yticks(range(len(pivot.index)), pivot.index)
            plt.xlabel('Crossover Rate')
            plt.ylabel('Mutation Rate')
            plt.title(f'Parameter Interaction Heatmap (Memory Length = {mem_length})')
            plt.savefig(f'grid_search_heatmap_mem{mem_length}.png')
            plt.close()
        
        print("Grid search visualizations saved.")
    def plot_noise_results(self, results_df):
        """Create visualizations of noise experiment results
    
    Args:
        results_df (pd.DataFrame): DataFrame with experiment results
    """
        import matplotlib.pyplot as plt
    
        # Plot noise level vs. best fitness for each memory length
        plt.figure(figsize=(10, 6))
    
        for mem_length in [1, 2]:
            subset = results_df[results_df['memory_length'] == mem_length]
            plt.plot(subset['noise_level'], subset['best_fitness'], 
                marker='o', linewidth=2, label=f'Memory-{mem_length}')
    
        plt.xlabel('Noise Level')
        plt.ylabel('Best Fitness Score')
        plt.title('Effect of Noise on Strategy Performance')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig('noise_vs_fitness.png')
    
        # Plot the performance gap between Memory-1 and Memory-2 at different noise levels
        plt.figure(figsize=(10, 6))
    
    # Group by noise level and calculate difference between Memory-2 and Memory-1
        noise_summary = results_df.pivot_table(
        index='noise_level', 
        columns='memory_length', 
        values='best_fitness'
        )
    
        performance_gap = noise_summary[2] - noise_summary[1]
        plt.bar(noise_summary.index, performance_gap, width=0.01)
    
        plt.xlabel('Noise Level')
        plt.ylabel('Performance Gap (Memory-2 - Memory-1)')
        plt.title('Memory-2 Advantage Over Memory-1 at Different Noise Levels')
        plt.grid(True, alpha=0.3)
        plt.savefig('memory_advantage_vs_noise.png')

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
    
    def run_noise_experiments(self, noise_levels=[0, 0.01, 0.05, 0.1, 0.2]):
        """Run experiments with different noise levels
    
    Args:
        noise_levels (List[float]): Noise levels to test
    
    Returns:
        pd.DataFrame: Results of experiments
    """
        results = []
    
        for noise in noise_levels:
            print(f"\nRunning experiment with noise level: {noise}")
        
        # Create GAs with Memory-1 and Memory-2
            ga_mem1 = IPDGeneticAlgorithm(
            pop_size=100,
            generations=50,
            mutation_rate=0.05,
            crossover_rate=0.8,
            memory_length=1,
            noise_level=noise
        )
        
            ga_mem2 = IPDGeneticAlgorithm(
            pop_size=100,
            generations=50,
            mutation_rate=0.05,
            crossover_rate=0.8,
            memory_length=2,
            noise_level=noise
        )
        
            # Evolve strategies
            best_genome_mem1, best_fitness_mem1 = ga_mem1.evolve()
            best_genome_mem2, best_fitness_mem2 = ga_mem2.evolve()
        
            # Save results
            results.append({
            'noise_level': noise,
            'memory_length': 1,
            'best_fitness': best_fitness_mem1,
            'final_avg_fitness': ga_mem1.avg_fitness_history[-1],
            'best_genome': best_genome_mem1
        })
        
            results.append({
            'noise_level': noise,
            'memory_length': 2,
            'best_fitness': best_fitness_mem2,
            'final_avg_fitness': ga_mem2.avg_fitness_history[-1],
            'best_genome': best_genome_mem2
        })
        
    # Convert to DataFrame and analyze
        results_df = pd.DataFrame(results)
    
    # Plot results
        self.plot_noise_results(results_df)
    
        return results_df

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
        """Play a game of Iterated Prisoner's Dilemma with communication noise

    Args:
        genome (List[int]): Strategy genome to evaluate
        opponent_strategy: Function that takes histories and returns a move
        rounds (int): Number of rounds to play

    Returns:
        Tuple[int, int]: (my_score, opponent_score)
    """
        my_history = []  # What I think opponent did
        opponent_history = []  # What opponent thinks I did
        actual_my_history = []  # What I actually did
        actual_opponent_history = []  # What opponent actually did
        my_score = 0
        opponent_score = 0

        for _ in range(rounds):
            # Get moves based on perceived history
            my_move = self.interpret_strategy(genome, my_history, actual_my_history)
            opponent_move = opponent_strategy(opponent_history, actual_opponent_history)
        
            # Record actual moves
            actual_my_history.append(my_move)
            actual_opponent_history.append(opponent_move)
        
            # Apply noise - moves might be misinterpreted
            perceived_my_move = my_move
            if random.random() < self.noise_level:
                perceived_my_move = 'D' if my_move == 'C' else 'C'
        
            perceived_opponent_move = opponent_move
            if random.random() < self.noise_level:
                perceived_opponent_move = 'D' if opponent_move == 'C' else 'C'
        
            # Update histories with what each player perceives
            opponent_history.append(perceived_my_move)  # What opponent thinks I did
            my_history.append(perceived_opponent_move)  # What I think opponent did
        
            # Update scores based on what actually happened
            payoff = self.payoff_matrix[(my_move, opponent_move)]
            my_score += payoff[0]
            opponent_score += payoff[1]

        return my_score, opponent_score



if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    
    # PART 1: Original grid search and evolution - UNCHANGED from your original code
    print("\n========== PART 1: EVOLUTION AGAINST FIXED STRATEGIES ==========")
    # Choose whether to run with default parameters or use grid search
    run_grid_search = True  # Set to False to skip grid search

    if run_grid_search:
        print("\nRunning Grid Search to find optimal parameters:")
        # Create a temporary instance to run the grid search
        temp_ga = IPDGeneticAlgorithm()  # Default noise_level=0
        best_params = temp_ga.grid_search()
        
        # Run with best parameters
        top_params = best_params.iloc[0]
        print("\nRunning final evolution with best parameters:")
        print(top_params)
        
        ga_best = IPDGeneticAlgorithm(
            pop_size=int(top_params['pop_size']),
            generations=100,  # Use more generations for final run
            mutation_rate=top_params['mutation_rate'],
            crossover_rate=top_params['crossover_rate'],
            memory_length=int(top_params['memory_length'])
        )
        
        best_genome, best_fitness = ga_best.evolve()
        ga_best.plot_performance()
    else:
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
    
    # PART 2: Run noise experiments
    print("\n========== PART 2: NOISE EXTENSION ==========")
    print("\nRunning Noise Experiments:")
    
    # Create a new instance for Part 2
    ga_part2 = IPDGeneticAlgorithm()  # Default noise level will be set in run_noise_experiments
    results_df = ga_part2.run_noise_experiments([0, 0.01, 0.05, 0.1, 0.2])
    
    # Save results to CSV
    results_df.to_csv('ipd_noise_experiment_results.csv', index=False)
    print("Noise experiment results saved to 'ipd_noise_experiment_results.csv'")
