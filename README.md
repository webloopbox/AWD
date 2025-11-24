## AWD – Data Analysis and Visualization (Heart & Wine)

This project delivers a complete workflow: statistical exploration and machine learning modeling (Random Forest, XGBoost) for two datasets:

- `winequality.csv` (quality of Portuguese red and white wines)
- `heart_cleveland_upload.csv` (cardiac parameters and heart disease presence)

### Structure

```
datasets/
  winequality.csv
  heart_cleveland_upload.csv
src/
  preprocessing/        # Normalization, encoding
  models/               # Model implementations (RandomForest, XGBoost)
  evaluation/           # Cross‑validation, metrics, hyperparameter tuning
  visualization/        # Plot generators (confusion matrix, ROC, learning curve)
  analysis/             # Statistical notebooks (wine, heart)
random_forest/
  random_forest.py
xgboost/
  xgboost.py
wine_statistics.ipynb   # (working / mirrored version in src/analysis)
```

Output folders:

- `wine_output/` – images, statistics CSV, final wine analysis summary
- `heart_output/` – analogous artifacts for heart dataset

### Requirements / Environment

Minimum versions (example):

- Python 3.10+
- pandas, numpy, seaborn, matplotlib
- scikit-learn, xgboost, scipy

Installation (example):

```bash
python -m venv .venv
source .venv/bin/activate  # macOS / Linux
pip install -r requirements.txt  # if generated
```

If `requirements.txt` is missing, install manually:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn xgboost scipy
```

### Data

Datasets should reside in the `datasets/` directory. Original column names are preserved; notebooks use a Polish label dictionary (`polish_labels`) to produce localized plots (easily adjustable to English labels if desired).

### Statistical Analysis

Notebooks in `src/analysis/` perform:

1. Data loading and quality checks (missing values, duplicates)
2. Descriptive statistics + CSV export
3. Correlations (matrix + heatmap)
4. Distributions: histograms, boxplots, violin plots
5. Relationship plots (scatter + simple linear regressions)
6. Feature ranking for wine quality / heart condition prediction
7. Textual summary (currently Polish – can be extended to English)

All generated figures and files are saved automatically to their respective output folders.

### Machine Learning Layer

Includes:

- Normalization / scaling (e.g., StandardScaler, MinMaxScaler)
- Categorical encoding (LabelEncoder wrapper)
- Random Forest and XGBoost baseline models
- Cross‑validation (e.g., 5-fold)
- Hyperparameter tuning (grid / random search)
- Metrics: accuracy, precision, recall, F1, ROC AUC
- Visualizations: confusion matrix, ROC curve, learning curve, feature importance

Example training snippet (inside an ML notebook):

```python
from models.random_forest_model import RandomForestWrapper
model = RandomForestWrapper(params={"n_estimators": 200, "max_depth": 8})
model.fit(X_train, y_train)
preds = model.predict(X_test)
```

### How to Run the Notebooks

1. Activate the virtual environment and install dependencies.
2. Open VS Code or Jupyter.
3. Execute cells sequentially in `src/analysis/wine_statistics.ipynb` or the heart statistics notebook.
4. Inspect `wine_output/` and `heart_output/` after completion.
