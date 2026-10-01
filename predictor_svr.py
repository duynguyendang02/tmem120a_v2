"""TMEM120A RBF-SVR prediction backend for the Streamlit webtool.

The final model uses the shared 931-feature RDKit + Mordred representation,
followed by the StandardScaler and RBF-SVR stored inside the fitted pipeline.
The feature-space OOD guard is inherited from the same 931-feature training
representation and is used only to suppress extreme out-of-distribution inputs.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
import os
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors
from scipy import sparse

StageCallback = Callable[[str, str], None]


@dataclass(frozen=True)
class Prediction:
    score: Optional[float]
    category: str
    canonical_smiles: str
    in_domain: bool
    max_abs_z: float
    ood_threshold: float
    nearest_training_similarity: float
    nearest_training_smiles: str
    nearest_training_docking_score: float
    model_name: str
    model_test_r2: float
    model_test_rmse: float
    model_test_mae: float
    model_test_pearson: float
    model_test_spearman: float


HERE = Path(__file__).resolve().parent
BUNDLED_MODEL_DIR = HERE / "model_assets" / "RBF_SVR_FINAL_FOR_WEBTOOL"
GOOGLE_DRIVE_MODEL_DIR = Path(
    "/content/drive/MyDrive/TMEM120A_Transformer_project/260903_QSAR/RBF_SVR_FINAL_FOR_WEBTOOL"
)


def _candidate_model_dirs() -> list[Path]:
    candidates: list[Path] = []
    env_dir = os.environ.get("RBF_SVR_MODEL_DIR")
    if env_dir:
        candidates.append(Path(env_dir).expanduser())
    candidates.extend([HERE, BUNDLED_MODEL_DIR, GOOGLE_DRIVE_MODEL_DIR, HERE / "RBF_SVR_FINAL_FOR_WEBTOOL"])
    return candidates


def find_model_dir() -> Path:
    required = {
        "SVR_RBF_931_FINAL.joblib",
        "preprocessing_parameters.joblib",
        "final_931_feature_manifest.csv",
        "training_reference_607.csv",
        "training_reference_morgan2048.npz",
        "feature_space_OOD_guard.npz",
        "model_metadata.json",
    }
    for path in _candidate_model_dirs():
        if path.is_dir() and all((path / name).exists() for name in required):
            return path
    searched = "\n".join(f"  - {p}" for p in _candidate_model_dirs())
    raise FileNotFoundError(
        "RBF-SVR model assets were not found. Set RBF_SVR_MODEL_DIR or place "
        "RBF_SVR_FINAL_FOR_WEBTOOL under model_assets.\nSearched:\n" + searched
    )


@lru_cache(maxsize=1)
def _assets():
    model_dir = find_model_dir()
    model = joblib.load(model_dir / "SVR_RBF_931_FINAL.joblib")
    prep = joblib.load(model_dir / "preprocessing_parameters.joblib")
    manifest = pd.read_csv(model_dir / "final_931_feature_manifest.csv")
    ref = pd.read_csv(model_dir / "training_reference_607.csv")
    ref_fp = sparse.load_npz(model_dir / "training_reference_morgan2048.npz").tocsr()
    guard = np.load(model_dir / "feature_space_OOD_guard.npz")
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))

    train_mean = np.asarray(guard["train_mean"], dtype=float)
    train_std = np.asarray(guard["train_std"], dtype=float)
    threshold = float(np.asarray(guard["threshold"]).reshape(-1)[0])

    if model.named_steps["scaler"].n_features_in_ != 931:
        raise ValueError("Saved StandardScaler does not expect 931 input features.")
    if list(manifest["feature_name"]) != list(prep["final_feature_names"]):
        raise ValueError("Feature manifest and preprocessing feature order do not match.")
    if train_mean.shape != (931,) or train_std.shape != (931,):
        raise ValueError("OOD guard does not match the 931-feature representation.")
    if ref_fp.shape[0] != len(ref):
        raise ValueError("Training fingerprint matrix and reference table have different row counts.")

    return model_dir, model, prep, ref, ref_fp, train_mean, train_std, threshold, metadata


def get_model_directory() -> str:
    return str(_assets()[0])


def _notify(callback: Optional[StageCallback], stage: str, state: str) -> None:
    if callback is not None:
        callback(stage, state)


def _parse_molecule(mol: Chem.Mol) -> tuple[Chem.Mol, str]:
    smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    parsed = Chem.MolFromSmiles(smiles)
    if parsed is None:
        raise ValueError("Could not parse the input molecular structure.")
    return parsed, Chem.MolToSmiles(parsed, canonical=True, isomericSmiles=True)


def _resolve_rdkit_descriptor(name: str, desc_map: dict):
    """Resolve a frozen RDKit descriptor name across RDKit versions.

    Some RDKit releases expose descriptors in Descriptors._descList while
    others expose the same calculation only as rdMolDescriptors.Calc<Name>.
    The trained model depends on descriptor semantics, not the registry path.
    """
    func = desc_map.get(name)
    if func is not None:
        return func

    # Direct attribute fallback.
    func = getattr(Descriptors, name, None)
    if callable(func):
        return func

    # RDKit C++ descriptor API fallback, e.g. NumAmideBonds -> CalcNumAmideBonds.
    func = getattr(rdMolDescriptors, f"Calc{name}", None)
    if callable(func):
        return func

    return None


def _rdkit_raw(mol: Chem.Mol, expected_names: list[str]) -> np.ndarray:
    desc_map = dict(Descriptors._descList)
    values = []
    missing = []

    for name in expected_names:
        func = _resolve_rdkit_descriptor(name, desc_map)
        if func is None:
            missing.append(name)
            values.append(np.nan)
            continue
        try:
            value = float(func(mol))
        except Exception:
            value = np.nan
        values.append(value)

    if missing:
        raise KeyError(
            "RDKit descriptor compatibility error. Missing calculations: "
            + ", ".join(missing[:20])
            + (" ..." if len(missing) > 20 else "")
        )

    return np.asarray(values, dtype=float)


def _mordred_raw(mol: Chem.Mol, expected_names: list[str]) -> np.ndarray:
    try:
        from mordred import Calculator, descriptors
    except Exception as exc:
        raise ImportError(
            "Mordred descriptors are required. Install the pinned mordred-ojmb dependency."
        ) from exc

    calc = Calculator(descriptors, ignore_3D=True)
    actual_names = [str(d) for d in calc.descriptors]
    if actual_names != list(expected_names):
        missing = [x for x in expected_names if x not in set(actual_names)][:10]
        raise RuntimeError(
            "Mordred descriptor order/version does not match the trained model. "
            f"Expected {len(expected_names)} descriptors, got {len(actual_names)}. "
            f"Example missing names: {missing}"
        )

    result = calc(mol)
    values = []
    for value in result:
        try:
            x = float(value)
            if not np.isfinite(x):
                x = np.nan
        except Exception:
            x = np.nan
        values.append(x)
    return np.asarray(values, dtype=float)


def _build_final_features(mol: Chem.Mol, prep: dict) -> np.ndarray:
    rd_raw = _rdkit_raw(mol, prep["rdkit_raw_descriptor_names"])
    rd = rd_raw[np.asarray(prep["rdkit_keep_mask"], dtype=bool)]
    rd_medians = np.asarray(prep["rdkit_train_medians"], dtype=float)
    rd = np.where(np.isfinite(rd), rd, rd_medians)

    mo_raw = _mordred_raw(mol, prep["mordred_raw_descriptor_names"])
    missing_mask = np.asarray(prep["mordred_missingness_keep_mask"], dtype=bool)
    valid_median_mask = np.asarray(prep["mordred_valid_median_mask"], dtype=bool)
    medians_all = np.asarray(prep["mordred_train_medians_before_variance_filter"], dtype=float)
    variance_mask = np.asarray(prep["mordred_variance_keep_mask"], dtype=bool)

    # All current saved missingness/median masks contain 1613 entries; applying them
    # explicitly preserves the frozen preprocessing logic and future-proofs the code.
    mo = mo_raw[missing_mask]
    medians = medians_all[missing_mask]
    vm = valid_median_mask[missing_mask]
    mo = mo[vm]
    medians = medians[vm]
    mo = np.where(np.isfinite(mo), mo, medians)
    variance_for_kept = variance_mask[missing_mask][vm]
    mo = mo[variance_for_kept]

    combined = np.concatenate([rd, mo])
    corr_mask = np.asarray(prep["combined_correlation_keep_mask"], dtype=bool)
    if combined.shape != corr_mask.shape:
        raise ValueError(
            f"Pre-correlation feature shape mismatch: {combined.shape} vs {corr_mask.shape}."
        )
    final_x = combined[corr_mask]
    if final_x.shape != (931,):
        raise ValueError(f"Expected 931 final descriptors, got {final_x.shape[0]}.")
    return final_x


def _nearest_training_analogue(mol: Chem.Mol, ref: pd.DataFrame, ref_fp: sparse.csr_matrix) -> dict:
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    fp = gen.GetFingerprint(mol)
    arr = np.zeros((2048,), dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    # Binary Morgan Tanimoto against the saved CSR reference matrix.
    query = sparse.csr_matrix(arr.reshape(1, -1))
    intersections = ref_fp.multiply(query).sum(axis=1).A1.astype(float)
    ref_counts = ref_fp.sum(axis=1).A1.astype(float)
    q_count = float(arr.sum())
    unions = ref_counts + q_count - intersections
    sims = np.divide(intersections, unions, out=np.zeros_like(intersections), where=unions > 0)
    idx = int(np.argmax(sims))
    row = ref.iloc[idx]
    return {
        "nearest_training_similarity": float(sims[idx]),
        "nearest_training_smiles": str(row["smiles"]),
        "nearest_training_docking_score": float(row["docking_score"]),
    }


def docking_category(score: float) -> str:
    if score < -6.45:
        return "VERY_GOOD"
    if score < -5.70:
        return "GOOD"
    if score < -4.945:
        return "MEDIUM"
    return "POOR"


def predict_docking_score(
    mol: Chem.Mol,
    target_name: str = "TMEM120A",
    stage_callback: Optional[StageCallback] = None,
) -> Prediction:
    if target_name.strip().upper() != "TMEM120A":
        raise ValueError("This trained QSAR model is specific to TMEM120A.")

    _, pipeline, prep, ref, ref_fp, train_mean, train_std, threshold, metadata = _assets()

    _notify(stage_callback, "structure", "processing")
    parsed, canonical = _parse_molecule(mol)
    _notify(stage_callback, "structure", "done")

    _notify(stage_callback, "descriptors", "processing")
    final_x = _build_final_features(parsed, prep)
    _notify(stage_callback, "descriptors", "done")

    _notify(stage_callback, "features", "processing")
    z = np.abs((final_x - train_mean) / train_std)
    max_abs_z = float(np.max(z))
    in_domain = bool(max_abs_z <= threshold)
    _notify(stage_callback, "features", "done")

    _notify(stage_callback, "standardize", "processing")
    scaler = pipeline.named_steps["scaler"]
    svr = pipeline.named_steps["svr"]
    x_scaled = scaler.transform(final_x.reshape(1, -1))
    _notify(stage_callback, "standardize", "done")

    _notify(stage_callback, "inference", "processing")
    raw_score = float(svr.predict(x_scaled).ravel()[0])
    nearest = _nearest_training_analogue(parsed, ref, ref_fp)
    _notify(stage_callback, "inference", "done")

    _notify(stage_callback, "report", "processing")
    if in_domain:
        score: Optional[float] = raw_score
        category = docking_category(raw_score)
    else:
        score = None
        category = "OUT_OF_DOMAIN"

    metrics = metadata["test_metrics"]
    prediction = Prediction(
        score=score,
        category=category,
        canonical_smiles=canonical,
        in_domain=in_domain,
        max_abs_z=max_abs_z,
        ood_threshold=threshold,
        nearest_training_similarity=float(nearest["nearest_training_similarity"]),
        nearest_training_smiles=str(nearest["nearest_training_smiles"]),
        nearest_training_docking_score=float(nearest["nearest_training_docking_score"]),
        model_name=str(metadata["model_name"]),
        model_test_r2=float(metrics["R2"]),
        model_test_rmse=float(metrics["RMSE"]),
        model_test_mae=float(metrics["MAE"]),
        model_test_pearson=float(metrics["Pearson"]),
        model_test_spearman=float(metrics["Spearman"]),
    )
    _notify(stage_callback, "report", "done")
    return prediction
