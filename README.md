# To run example scripts 
First, cd to the `proteingym` folder and run `pip install -e .`. After that, running `scripts/run_model.sh` or `run_ensemble.sh`
should work to run the two example scripts. 

# Library Structure/Adding a New Model 
The scoring (and eventually statistics) scripts just rely on each model subclassing the `SequenceFitnessModel` class 
in `proteingym/wrappers/generic_models.py` and then implementing the `predict_fitnesses` function. So far I've added 
ESM2, Tranception (without retrieval), a site-independent model and MSA transformer as example models. Ideally, we eventually write all models into this format, and then have a single script (currently at `scripts/run_model.py`) that will score any model on any of the ProteinGym datasets. There's also a class for an example ensemble (`SequenceFitnessModelEnsemble` in `proteingym/wrappers/model_ensembles.py`) to show that, with a uniform interface, we can combine a whole bunch of models together and write really simple ensembling abstractions. 

The configurations for each model are defined by json config files, see the `example_configs` folder, that specify specific parameters for each model. Those parameters are determined by the model class itself, e.g. the TranceptionModel class, and also the relevant parent classes (ProteinLanguageModel and SequenceFitnessProbabilityModel for Tranception). Writing a more detailed configuration description is on my to-do list, but the ones in `example_configs` are illustrative. 

# Documentation 
So far almost all functions and some classes have docstrings in the Google docstring format. I'm going to go add the rest sometime soon. If all the classes and functions have uniform docstrings in that format, we can autogenerate documentation using Sphinx and host it somewhere, if that would be useful. 

# Outstanding Things to Add (likely an incomplete list)
* Structure models, e.g. ProteinMPNN 
* The remaining zero-shot models 
* Code for fine-tuning and supervised models 
* Logging and writing scores to output files (can be same as current ProteinGym)
* statistics script for computing spearmans and other metrics from score files (can also be the same as the current ProteinGym one if the output file formats are the same)
* Additional missing documentation for any functions 