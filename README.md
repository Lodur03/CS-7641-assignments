# CS 7641 Assignments

If a python virtual environment has been setup for the project, a simple `pip install -r requirements.txt` should take care of the required packages.

Running `python run_experiment.py -h` (in each assignment folder) should provide a list of options for what you can do.

For the most part it is simple to run a given set of experiments based on a specific algorithm. One flag to consider always including is `--threads` with a value of `-1`. This will speed up execution in some cases but also might use all available cores.

The `--verbose` flag can be helpful to view data about a given dataset or MDP.

For assignments 3 and 4 plotting data is a separate step from generation. For those assignments the `--plot` flag should be used once data is generated

Each assignment folder should have its own readme with anything specific to not for that assignment.
