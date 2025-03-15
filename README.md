# Iterated Prisoner's Dilemma - Genetic Algorithm (CT421 Project 2)

## Overview
This project implements a genetic algorithm to evolve strategies for the Iterated Prisoner's Dilemma (IPD) game. The code consists of two main parts:
1. **Part 1**: Evolution against fixed strategies
2. **Part 2**: Extension with communication noise

## Requirements
- Python 3.x
- Required packages: 
  - matplotlib
  - pandas
  - numpy (implicit dependency)

## How to Run

Simply execute the main script:
```
python genetic_algorithm.py
```

The code will run both Part 1 and Part 2 sequentially. A random seed (42) is set at the beginning to ensure reproducible results.

## Code Structure & Flow

1. **Initialization**: The genetic algorithm is initialized with configurable parameters
2. **Part 1**: Runs grid search to find optimal parameters, then evolves strategies against fixed opponents
3. **Part 2**: Runs noise experiments to evaluate the effect of communication errors on strategy performance

## Part 1: Evolution Against Fixed Strategies

In Part 1, the code:
1. Performs a grid search across multiple parameter combinations:
   - Population sizes: 50, 100
   - Mutation rates: 0.01, 0.05, 0.1
   - Crossover rates: 0.7, 0.8, 0.9
   - Memory lengths: 1, 2

2. Selects the best parameter combination and runs a full evolution (100 generations)

3. Outputs:
   - `grid_search_mutation_rate.png`: Effect of mutation rate on fitness
   - `grid_search_memory_length.png`: Impact of memory length on performance
   - `grid_search_heatmap_mem1.png` and `grid_search_heatmap_mem2.png`: Parameter interaction heatmaps
   - `ipd_evolution_performance.png`: Performance over generations
   - `ipd_grid_search_results.csv`: Detailed results from grid search

## Part 2: Noise Extension

Part 2 investigates how communication noise affects strategy performance:
1. Runs experiments with different noise levels (0, 0.01, 0.05, 0.1, 0.2)
2. Tests both Memory-1 and Memory-2 strategies at each noise level
3. Outputs:
   - `noise_vs_fitness.png`: Effect of noise on strategy performance
   - `memory_advantage_vs_noise.png`: Performance gap between Memory-2 and Memory-1
   - `ipd_noise_experiment_results.csv`: Detailed results from noise experiments

## Strategy Representation

- **Memory-1 strategies**: 3 genes (first move, response to C, response to D)
- **Memory-2 strategies**: 5 genes (first move, response to CC, CD, DC, DD)

## Fixed Strategies

Evolved strategies are tested against:
- Always Cooperate
- Always Defect
- Tit-for-Tat (cooperate first, then copy opponent's last move)
- Suspicious Tit-for-Tat (defect first, then copy opponent's last move)

## Configuration Options

In the main script, you can modify:
- `run_grid_search = True/False`: Enable/disable the grid search (skipping directly to evolution)
- Parameters in the evolution runs (population size, generations, mutation rate, etc.)
- Noise levels tested in Part 2

## Output Files

The program generates:
1. CSV files for data analysis
2. PNG visualizations showing performance metrics
3. Terminal output for tracking progress

## Implementation Details

- Tournament selection is used for parent selection
- Single-point crossover for genetic recombination 
- Bit-flip mutation
- Elitism (preserves best solutions)
- Strategies are evaluated by their average score against all fixed strategies

## Noise Implementation

Noise is implemented as a probability of a move being misinterpreted by the opponent. When noise occurs, a "C" is perceived as a "D" or vice versa. This simulates communication errors in real-world interactions.