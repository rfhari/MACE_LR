## Setup

Follow instructions to install MACE. Then install following packages

```
pip install torchmetrics
pip install h5py
pip install torch_geometric
pip install hostlist
```

## Testing with single water molecule (with and without pbc)

* Test notebook in 'mace/MACE_developer_hariharr.ipynb'
* Modified model files
    * ['models_hariharr_energy_ewald.py'](mace/mace/modules/models_hariharr_energy_ewald.py): main script with MACE and Ewald block
    * ['models_hariharr_dipole.py'](mace/mace/modules/models_hariharr_dipole.py): dipole block for charges: discuss later 
