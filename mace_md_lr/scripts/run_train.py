## Wrapper for mace.cli.run_train.main ##

# import sys
# import os


# custom_module_path = '/home/hari/Desktop/Research/MACE_LR_train/github_files/MACE_LR/mace'
# sys.path.insert(0, custom_module_path)

from mace.cli import run_train
from mace.cli.run_train import main
print(f"run_train imported from: {run_train.__file__}")

if __name__ == "__main__":
    main()
    # print(f"my_module imported from: {main.__file__}")

