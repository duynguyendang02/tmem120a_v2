# TMEM120A Molecular Screening Platform — RBF-SVR backend

Streamlit webtool for rapid TMEM120A docking-score estimation and molecular property analysis.

## Scientific model

- Input: molecular structure / SMILES
- Descriptor engines: RDKit 2D + Mordred 2D
- Pre-correlation descriptor pool: 1,526
- Final retained descriptors: 931
- Preprocessing: fitted StandardScaler
- Model: RBF-SVR (`C=8`, `gamma=0.0003`, `epsilon=0.40`)
- Training set: 607 compounds
- Independent test: 202 compounds
- Test R2: 0.6357
- Test RMSE: 0.5752 kcal/mol
- Test MAE: 0.4624 kcal/mol
- Test Spearman: 0.6864

The QSAR score is intended for rapid prioritization and is not a substitute for explicit molecular docking or experimental validation.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The bundled model package is under `model_assets/RBF_SVR_FINAL_FOR_WEBTOOL/`.

## Deployment

Push the repository to GitHub and redeploy/reboot the Streamlit app. If Git LFS or host file-size limits become relevant, the backend also supports the `RBF_SVR_MODEL_DIR` environment variable for an external model-assets directory.
