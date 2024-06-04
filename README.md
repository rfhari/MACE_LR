## Setup

Follow instructions to install MACE. 

```
pip install --upgrade pip
pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cu118
pip install mace-torch
```

Then install following packages

```
pip install torchmetrics
pip install h5py
pip install torch_geometric
pip install hostlist
```

## Testing with single water molecule (with and without pbc)

* Modified model files
    * ['models.py'](mace_train/mace/modules/models.py): main script with MACE and Ewald block
    
