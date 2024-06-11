## Usage

* MACE + Ewald scripts within mace_train folder
* Example python cli to train

```
python ./scripts/run_train.py --name="MACE_Ewald" --num_workers=16 --forces_key='forces' --r_max=3.0 --energy_key='Energy' --train_file="../custom_data/md22_buckyball_catcher/buckyball_catcher_train.xyz" --valid_fraction=0.05 --test_file="../custom_data/md22_buckyball_catcher/buckyball_catcher_test.xyz" --model="MACE_Ewald" --num_interactions=2 --num_channels=256 --E0s="average" --max_L=2 --correlation=3 --batch_size=8 --valid_batch_size=8 --max_num_epochs=650 --swa --start_swa=450 --ema --ema_decay=0.99 --amsgrad --forces_weight=1000 --energy_weight=10 --scheduler_patience=5 --patience=15 --swa_forces_weight=10 --error_table='PerAtomMAE' --device=cuda --seed=123 --restart_latest > bucky_trail.out
```

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
    
