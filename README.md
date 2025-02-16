# Evolutionary Search Project 1 for CT421

## Overview
Solving the Traveling Salesman Problem using a genetic algorithm for Assignment 1.

## Authors
- Tim Samoska: 21326923
- Alasdair Ball: 21436934

## Installation
To install the necessary dependencies, run:
```bash
pip install -r requirements.txt
```

## Usage/Setup
### User Defined Parameters
At the start of the main function, there is a user-defined parameters section.
The following can be modified:
- DATASET_PATH: Path to where the TSP files are stored.
- PROBLEM_FILE: TSP file name to test for this run.
- GEN_LIMIT: The generation limit the algorithm will run.  
             (Provided if the algorithm has not stagnated.)
#### Grid Search Variables
- pop_sizes: Range of population size variables to test.
- crossover_rates: Range of crossover chances to test.
- mutation_rates: Range of mutation chances to test.

### Run
To run the evolutionary search, execute within the EvolutionarySearch directory:
```bash
python geneticAlgorithm.py
```

## Linting
Linted with Microsoft's PyLint
