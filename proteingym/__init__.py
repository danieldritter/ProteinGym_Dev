"""
this import is necessary to import all the model classes 
in the proteingym.models package (see models/__init__.py for the import logic), 
and each of those models automatically adds its constructor to an _available_models 
dictionary in the SequenceFitnessModel class
"""
import proteingym.models