"""
Molecular property prediction pipeline.

Featurizes SMILES with RDKit descriptors + Morgan fingerprints, does
scaffold-based GroupKFold cross-validation, and trains an XGBoost
regressor per fold.

Usage:
    python mol_predict.py --input data.csv --smiles-col smiles --target-col y
"""

import sys

import click
import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold
from xgboost import XGBRegressor

DESCRIPTOR_FUNCS = [
    Descriptors.MolWt,
    Crippen.MolLogP,
    Descriptors.TPSA,
    Lipinski.NumHDonors,
    Lipinski.NumHAcceptors,
    Lipinski.NumRotatableBonds,
    Lipinski.RingCount,
]


def featurize_smiles(smiles, radius=2, n_bits=2048):
    """Convert a SMILES string into a descriptor + Morgan fingerprint vector.

    Returns None if the SMILES cannot be parsed.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    descriptors = [func(mol) for func in DESCRIPTOR_FUNCS]
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        mol, radius=radius, nBits=n_bits
    )
    return np.array(descriptors + list(fingerprint), dtype=np.float32)


def get_scaffold(smiles):
    """Return the Murcko scaffold SMILES for a molecule, or '' on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=mol, includeChirality=False)


def build_dataset(df, smiles_col, target_col, radius, n_bits):
    """Featurize all rows, dropping any that fail to parse."""
    features = []
    valid_indices = []

    for idx, smiles in enumerate(df[smiles_col]):
        x = featurize_smiles(smiles, radius=radius, n_bits=n_bits)
        if x is not None:
            features.append(x)
            valid_indices.append(idx)

    n_dropped = len(df) - len(valid_indices)
    if n_dropped:
        click.echo(f"Dropped {n_dropped} unparseable SMILES.")

    if not features:
        raise click.ClickException("No valid SMILES could be featurized.")

    X = np.vstack(features)
    df_valid = df.iloc[valid_indices].reset_index(drop=True)
    y = df_valid[target_col].values
    return X, y, df_valid


def run_cv(X, y, groups, n_splits, xgb_params):
    """Run scaffold-grouped K-fold CV, training one XGBRegressor per fold."""
    cv = GroupKFold(n_splits=n_splits)
    fold_results = []

    for fold, (train_idx, valid_idx) in enumerate(cv.split(X, y, groups=groups)):
        X_train, X_valid = X[train_idx], X[valid_idx]
        y_train, y_valid = y[train_idx], y[valid_idx]

        click.echo(
            f"Fold {fold}: train={len(train_idx)}, valid={len(valid_idx)}"
        )

        model = XGBRegressor(**xgb_params)
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_valid, y_valid)],
            verbose=False,
        )

        pred = model.predict(X_valid)
        rmse = float(np.sqrt(mean_squared_error(y_valid, pred)))
        mae = float(mean_absolute_error(y_valid, pred))
        r2 = float(r2_score(y_valid, pred))

        fold_results.append({"fold": fold, "RMSE": rmse, "MAE": mae, "R2": r2})

    return pd.DataFrame(fold_results)


@click.command()
@click.option(
    "--input", "-i", "input_path",
    required=True,
    type=click.Path(exists=True, dir_okay=False),
    help="Path to input CSV file.",
)
@click.option(
    "--smiles-col", default="smiles", show_default=True,
    help="Name of the column containing SMILES strings.",
)
@click.option(
    "--target-col", default="y", show_default=True,
    help="Name of the column containing the regression target.",
)
@click.option(
    "--n-splits", default=5, show_default=True,
    help="Number of scaffold-grouped CV folds.",
)
@click.option(
    "--radius", default=2, show_default=True,
    help="Morgan fingerprint radius.",
)
@click.option(
    "--n-bits", default=2048, show_default=True,
    help="Morgan fingerprint bit-vector length.",
)
@click.option(
    "--n-estimators", default=1000, show_default=True,
    help="Number of XGBoost trees.",
)
@click.option(
    "--max-depth", default=6, show_default=True,
    help="Max tree depth.",
)
@click.option(
    "--learning-rate", default=0.03, show_default=True,
    help="XGBoost learning rate.",
)
@click.option(
    "--subsample", default=0.8, show_default=True,
    help="Row subsample ratio.",
)
@click.option(
    "--colsample-bytree", default=0.8, show_default=True,
    help="Column subsample ratio per tree.",
)
@click.option(
    "--seed", default=42, show_default=True,
    help="Random seed.",
)
@click.option(
    "--output", "-o", "output_path",
    default=None,
    type=click.Path(dir_okay=False),
    help="Optional path to save per-fold results as CSV.",
)
def main(
    input_path,
    smiles_col,
    target_col,
    n_splits,
    radius,
    n_bits,
    n_estimators,
    max_depth,
    learning_rate,
    subsample,
    colsample_bytree,
    seed,
    output_path,
):
    """Featurize SMILES, run scaffold-split CV, and train XGBoost regressors."""
    df = pd.read_csv(input_path)

    for col in (smiles_col, target_col):
        if col not in df.columns:
            raise click.ClickException(
                f"Column '{col}' not found in {input_path}. "
                f"Available columns: {list(df.columns)}"
            )

    click.echo(f"Loaded {len(df)} rows from {input_path}")

    X, y, df_valid = build_dataset(df, smiles_col, target_col, radius, n_bits)
    click.echo(f"Feature matrix: {X.shape}")

    click.echo("Computing Murcko scaffolds...")
    df_valid["scaffold"] = df_valid[smiles_col].apply(get_scaffold)
    groups = df_valid["scaffold"].values

    n_groups = len(set(groups))
    if n_groups < n_splits:
        raise click.ClickException(
            f"Only {n_groups} unique scaffolds found, but --n-splits={n_splits}. "
            "Reduce --n-splits or provide more diverse data."
        )

    xgb_params = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        subsample=subsample,
        colsample_bytree=colsample_bytree,
        objective="reg:squarederror",
        eval_metric="rmse",
        random_state=seed,
        n_jobs=-1,
    )

    results = run_cv(X, y, groups, n_splits, xgb_params)

    click.echo("\nPer-fold results:")
    click.echo(results.to_string(index=False))

    click.echo("\nMean:")
    click.echo(results[["RMSE", "MAE", "R2"]].mean().to_string())

    if output_path:
        results.to_csv(output_path, index=False)
        click.echo(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    main()