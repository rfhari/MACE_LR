## Setup

Follow instructions to install MACE. Then install following packages

```
pip install torchmetrics
pip install h5py
pip install torch_geometric
pip install hostlist
```

## Testing with single water molecule (with and without pbc)

* Modified model files
    * ['models.py'](mace_train/mace/modules/models.py): main script with MACE and Ewald block
    
