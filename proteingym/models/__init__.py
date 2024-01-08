"""
This is a super hacky way to do things. Essentially we add the model_repos folder 
to the system path, which includes all the packages necessarily for the baselines. 
This allows the relative imports within those packages to work correctly. 
After that we remove the model_repos folder from the system path, along with the 
models folder path
"""
import pkgutil
import importlib
import sys
import os
from pathlib import Path

package_name = "proteingym.models"
package = sys.modules[package_name]

# Add the package directory to sys.path
package_path = Path(package.__file__).resolve().parent
sys.path.append(str(package_path))
# Add the model_repo subdirectories to sys.path
model_repo_path = str(package_path) + os.sep + "model_repos"
sys.path.append(str(model_repo_path))

for loader, module_name, is_pkg in pkgutil.walk_packages(package.__path__):
    if not loader.path.endswith("models"):
        continue
    _module = importlib.import_module(module_name)

# Remove the package directory from sys.path
sys.path.remove(str(package_path))
sys.path.remove(str(model_repo_path))
