## Changelog for CT421 Project 1 Evolutionary Search
### 9/2/25
#### Tim
- Created readme and changelog markdown files
- Split the provided TSP Dataset file into 3 separate TSP files in tsp_datasets
- Removed the temp test TSP file
- Created a basic starting point for the readme
- Fixed spelling mistake in genetic algorithm file name
- Improved folder access if the tsp files, now can be accessed directly if the search is run in the src folder or outside it
- Fixed the plot function, now the performance is plotted properly
- Now printing the best path to console
- Removed redundant function in tsp loader

### 14/2/25
#### Tim
- Removed redundant comments
- Added function doc strings
- Applied linting (Pylint) across genetic_algorithm.py
- Added crossover rate

### 15/2/25
#### Tim
- Added ability to grid search a range of parameters
- Multiple runs are now made
- Results are output to a csv file
- Refactoring of main entry function
- Integrated TSPLoader aspects into main function
- Optimization and logic fixes
- Implemented a chance for Inversion swapping to be used