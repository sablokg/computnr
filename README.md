# compnutr

- Computation precision Nutrition ML
- Applied XGBoost to the chemical smiles coming from the food chemical data and boosting the scaffold approach.
- Only two types of ML is possible and hold importance in computational precise nutrition: 1. Classical ML and 2. Graph ML https://arxiv.org/abs/1703.00564

```
pip install rdkit xgboost click scikit-learn pandas numpy
python mol_predict.py --input data.csv --smiles-col smiles --target-col y --n-splits 5 -o results.csv

```

Gaurav Sablok \
gsablok@proton.me