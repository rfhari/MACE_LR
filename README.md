## Setup

if you want to run MACE-tutorial notebook, clone MACE-dev version from Nov 17 2023 (this might not be relevant now since we are running just the forward)
#TO DO: should update to latest version and try

```
git clone https://github.com/ACEsuit/mace
git checkout ed2c3ba45a9e4f7d2f632967f02c114ec9e374f0
git status #just to verify the version
```

## Testing with single water molecule (with and without pbc)

* Test notebook in 'mace/MACE_developer_hariharr.ipynb'
* Modified model files
    * ['models_hariharr_energy_ewald.py'](mace/mace/modules/models_hariharr_energy_ewald.py): main script with MACE and Ewald block
    * ['models_hariharr_dipole.py'](mace/mace/modules/models_hariharr_dipole.py): dipole block for charges: discuss later 
