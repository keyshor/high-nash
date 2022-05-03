# Dependencies

1. Requires Python version 3.7 or higher. Requirements can be installed using:

```
pip install -r requirements.txt
```

2. Requires Spot (https://spot.lrde.epita.fr/). 

Installation instructions can be found at https://spot.lrde.epita.fr/install.html. 

3. Requires Gambit (https://gambitproject.readthedocs.io/en/latest/intro.html)


# Running the code

To run our algorithm on the Intersection environment, use the command

```
python -m spectrl.examples.intersection -n <run_number> -d <directory> -s <spec_number>
```

 `<run_number>` is an integer (can vary from 0-9) used to distinguish different runs for the same specification. `<spec_number>` is either 0, 1, 2, 3 or 4 corresponding to specs 1 to 5 in the paper. Results are stored in  `<directory>/spec<spec_number>/search`. Please use different directories for different environments, but use same directory for all specs and runs corresponding to an environment.

Similarly, for the Single Lane environment, the command is

```
python -m spectrl.examples.sortworld -n <run_number> -d <directory> -s <spec_number>
```

`<spec_number>` is either 0, 1, 2, 3, 6, 7 corresponding to specs 1 to 6 in the paper.

For the Gridworld environment, the command is

```
python -m spectrl.examples.gridworld -n <run_number> -d <directory> -s <spec_number>
```

Here `<spec_number>` is either 0, 2, 3, 4 or 5 corresponding to specs 1 to 5 in the paper. 

Append `_nvi` to the above commands (to the module name, e.g., `spectrl.examples.intersection_nvi`) to run NVI baseline and append `_maqrm` to run MAQRM baseline. Results are stored in  `<directory>/spec<spec_number>/nvi` or  `<directory>/spec<spec_number>/multi_qrm` depending on the algorithm.

# Printing final results

After running any algorithm on a specific benchmark 10 times (different runs can run in parallel using different values for -n option), use the following command to output results to console.

```
python -m spectrl.examples.multi_eval -n 10 -d <directory>/spec<spec_number>
```

Prints the average values of social welfare and nash deviations of the trained policies across runs. Here `<directory>` corresponds to the directory where the results are stored for the particular environment and `<spec_number>` is the specification numbers mentioned in the previous section.