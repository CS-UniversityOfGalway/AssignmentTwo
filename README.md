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
             (Provided if the algorithm has not stagnated in the last 100 generations.)
#### Grid Search Variables
- pop_sizes: Range of population size variables to test.
- crossover_rates: Range of crossover chances to test.
- mutation_rates: Range of mutation chances to test.

### Example
Running the code as is with no modifcations, will do 1000 generation grid search test run on the pr1002.tsp dataset is the tsp_datasets folder, with the pop_sizes 200, 225, 250,crossover_rates 0.7, 0.8, 0.9 and mutation_rates 0.01, 0.02, 0.05.

### Run
To run the evolutionary search, execute within the EvolutionarySearch directory:
```bash
python genetic_algorithm.py
```

### Output
After the run is complete, the following will happen:
- A graph will appear for the best run, with best fitness plotted against generations.
- CMD output of the results of the best run and time taken.
- A CSV file will be output with the results for the entire grid search test run. 
## Linting
Linted with Microsoft's PyLint
