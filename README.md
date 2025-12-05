# CryoFlow: Amortized pose inference for Cryo-EM using a normalizing flow-based pose representation

## Requirements

To install requirements in your current environment:

```setup
pip install -r requirements.txt
pip install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
```

To replicate results, first generate the synthetic dataset using:

```bash
python -m src.reconstruct.main -c mrc2star_spliceosome.ini
```

Run to replicate convergence results:

```bash
bash run_experiments.sh configfiles/train_spliceosome_mode.ini nobeta --beta 0
bash run_experiments.sh configfiles/train_spliceosome_mode.ini beta --beta 2e-5
bash run_experiments.sh configfiles/train_spliceosome_cryo.ini cryo
```

Run to replicate reconstruction results:
```bash
python -m src.reconstruct.main -c configfiles/train_spliceosome_mode.ini --experiment_name longbeta --beta 2e-5 --train_epochs 60
python -m src.reconstruct.main -c configfiles/train_spliceosome_mode.ini --experiment_name longnobeta --beta 0 --train_epochs 60
python -m src.reconstruct.main -c configfiles/train_spliceosome_cryo.ini --experiment_name longcryo --train_epochs 60
```

Degree defect experiments can be found in `degree_defects.ipynb`. 