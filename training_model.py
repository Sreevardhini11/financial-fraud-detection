import os
import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -----------------------------
# Config
# -----------------------------
DATASETS = {
    "Credit Card":      "datasets/creditcard.csv",
    "Insurance Claims": "datasets/insurance_data.csv",
    "Loan Application": "datasets/loan_applications.csv",
    "Online Payment":   "datasets/onlinefraud.csv",
}

# If you KNOW a dataset's target label, set it here to override auto-detect.
TARGETS_OVERRIDE = {
    "Loan Application": "fraud_flag",   # adjust if your dataset has a different fraud label
    "Online Payment": "isFraud",        # common name in online fraud datasets
}

# Candidate target names to try if override not found
TARGET_CANDIDATES = [
    "Class", "Fraudulent", "isFraud", "label", "Label", "target", "Target", "y", "fraud_flag"
]

# Fraction of rows to train on (speed-up)
SAMPLE_FRAC = 0.2  # set to 1.0 for full dataset
MIN_SAMPLES_PER_CLASS = 10  # Minimum samples per class after sampling

# Ensure model directory exists
os.makedirs("models", exist_ok=True)

# Handle sklearn version differences for OneHotEncoder
sk_ver = tuple(map(int, sklearn.__version__.split(".")[:2]))
if sk_ver >= (1, 2):
    OHE = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
else:
    OHE = OneHotEncoder(handle_unknown="ignore", sparse=False)


# -----------------------------
# Helpers
# -----------------------------
def detect_target_column(df: pd.DataFrame, dataset_name: str) -> str | None:
    """Find a reasonable target/fraud column automatically."""
    # 1) Respect override
    if dataset_name in TARGETS_OVERRIDE and TARGETS_OVERRIDE[dataset_name] in df.columns:
        return TARGETS_OVERRIDE[dataset_name]

    # 2) Exact candidate names
    for name in TARGET_CANDIDATES:
        if name in df.columns:
            return name

    # 3) Case-insensitive matches
    lower_map = {c.lower(): c for c in df.columns}
    for name in TARGET_CANDIDATES:
        if name.lower() in lower_map:
            return lower_map[name.lower()]

    # 4) Heuristic: binary columns
    bin_cols = []
    for c in df.columns:
        vc = df[c].dropna().unique()
        if len(vc) <= 2:
            norm = set([str(x).strip().lower() for x in vc])
            if norm.issubset({"0", "1", "true", "false"}):
                bin_cols.append(c)

    # Prefer names with fraud-like keywords
    priority = ["fraud", "class", "label", "target", "y"]
    for key in priority:
        for c in bin_cols:
            if key in c.lower():
                return c

    # fallback
    return bin_cols[0] if bin_cols else None


def coerce_numeric_strings(X: pd.DataFrame) -> pd.DataFrame:
    """Convert object columns that are mostly numeric into numeric dtype."""
    Xc = X.copy()
    for col in Xc.select_dtypes(include=["object"]).columns:
        s = pd.to_numeric(Xc[col], errors="coerce")
        orig_non_null = Xc[col].notna()
        if orig_non_null.sum() == 0:
            continue
        ok_ratio = s.notna().sum() / orig_non_null.sum()
        if ok_ratio >= 0.8:
            Xc[col] = s
    return Xc


def check_class_balance(y: pd.Series, dataset_name: str) -> bool:
    """Check if we have enough samples per class for meaningful training."""
    class_counts = y.value_counts()
    print(f"   Class distribution: {dict(class_counts)}")
    
    if len(class_counts) < 2:
        print(f"❌ Only {len(class_counts)} class found in {dataset_name}")
        return False
    
    min_count = class_counts.min()
    if min_count < MIN_SAMPLES_PER_CLASS:
        print(f"❌ Minimum class has only {min_count} samples (need at least {MIN_SAMPLES_PER_CLASS})")
        return False
    
    return True


def stratified_sample(df: pd.DataFrame, target_col: str, frac: float, random_state: int = 42) -> pd.DataFrame:
    """Perform stratified sampling while ensuring minimum samples per class."""
    if target_col not in df.columns:
        return df.sample(frac=frac, random_state=random_state)
    
    # Check original class distribution
    orig_counts = df[target_col].value_counts()
    print(f"   Original class distribution: {dict(orig_counts)}")
    
    # If any class has fewer samples than our minimum after sampling, adjust strategy
    min_samples_needed = MIN_SAMPLES_PER_CLASS / frac
    
    if orig_counts.min() < min_samples_needed:
        print(f"   ⚠️  Some classes too small for stratified sampling, using simple sampling")
        return df.sample(frac=frac, random_state=random_state)
    
    # Perform stratified sampling
    sampled = df.groupby(target_col, group_keys=False).apply(
        lambda x: x.sample(frac=frac, random_state=random_state) if len(x) > 0 else x
    ).reset_index(drop=True)
    
    return sampled


# -----------------------------
# Training loop
# -----------------------------
successful_models = []
failed_models = []

for name, path in DATASETS.items():
    print(f"\n🚀 Training model for {name}...")

    try:
        # Load dataset
        if not os.path.exists(path):
            print(f"❌ File not found: {path}. Skipping {name}.")
            failed_models.append((name, "File not found"))
            continue

        df = pd.read_csv(path)
        if df.empty:
            print(f"❌ {name} dataset is empty. Skipping.")
            failed_models.append((name, "Empty dataset"))
            continue

        print(f"   Dataset shape: {df.shape}")

        # Detect target
        target_col = detect_target_column(df, name)
        if not target_col:
            print(f"❌ Could not detect target column for {name}. "
                  f"Columns: {list(df.columns)[:10]} ... Skipping.")
            failed_models.append((name, "No target column found"))
            continue
        print(f"✅ Using target column: {target_col}")

        # Optional sampling (stratified if target exists)
        if SAMPLE_FRAC < 1.0:
            df = stratified_sample(df, target_col, SAMPLE_FRAC, random_state=42)
            print(f"📉 Using {len(df)} rows ({int(SAMPLE_FRAC*100)}%) for training.")

        if target_col not in df.columns:
            print(f"❌ Target column '{target_col}' missing after sampling. Skipping {name}.")
            failed_models.append((name, "Target column missing after sampling"))
            continue

        X_raw = df.drop(columns=[target_col])
        y = df[target_col]

        # Check class balance
        if not check_class_balance(y, name):
            failed_models.append((name, "Insufficient class balance"))
            continue

        # Clean features
        X = coerce_numeric_strings(X_raw)
        
        # Remove columns with too many missing values (>90%)
        missing_thresh = 0.9
        missing_ratios = X.isnull().mean()
        high_missing_cols = missing_ratios[missing_ratios > missing_thresh].index.tolist()
        if high_missing_cols:
            print(f"   Dropping {len(high_missing_cols)} columns with >90% missing values")
            X = X.drop(columns=high_missing_cols)

        if X.empty or len(X.columns) == 0:
            print(f"❌ No features left after cleaning for {name}")
            failed_models.append((name, "No features after cleaning"))
            continue

        numeric_cols = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
        categorical_cols = X.select_dtypes(exclude=[np.number, "bool"]).columns.tolist()
        
        print(f"   Features: {len(numeric_cols)} numeric, {len(categorical_cols)} categorical")

        # Build preprocessing - only include transformers for existing column types
        transformers = []
        if numeric_cols:
            numeric_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="mean")),
                ("scaler", StandardScaler())
            ])
            transformers.append(("num", numeric_transformer, numeric_cols))

        if categorical_cols:
            categorical_transformer = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OHE)
            ])
            transformers.append(("cat", categorical_transformer, categorical_cols))

        if not transformers:
            print(f"❌ No valid transformers for {name}")
            failed_models.append((name, "No valid feature types"))
            continue

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False
        )

        # Build model pipeline
        model = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=50,
                n_jobs=-1,
                random_state=42,
                class_weight='balanced'  # Handle imbalanced classes
            ))
        ])

        # Train/test split
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        except ValueError as e:
            print(f"❌ Train/test split failed for {name}: {e}")
            # Try without stratification
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            print(f"   Using non-stratified split")

        # Train model
        print(f"   Training on {len(X_train)} samples...")
        model.fit(X_train, y_train)

        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Save model
        model_filename = f"models/{name.lower().replace(' ', '_')}_model.pkl"
        joblib.dump(model, model_filename)

        # Report results
        print(f"✅ {name} model trained successfully!")
        print(f"   Target column: {target_col}")
        print(f"   Training accuracy: {train_score:.4f}")
        print(f"   Test accuracy: {test_score:.4f}")
        print(f"   Model saved to: {model_filename}")
        
        successful_models.append(name)

    except Exception as e:
        print(f"❌ Unexpected error training {name}: {str(e)}")
        failed_models.append((name, f"Unexpected error: {str(e)}"))
        continue

# Summary
print("\n" + "="*50)
print("TRAINING SUMMARY")
print("="*50)
print(f"✅ Successfully trained: {len(successful_models)} models")
for model in successful_models:
    print(f"   - {model}")

if failed_models:
    print(f"\n❌ Failed to train: {len(failed_models)} models")
    for model, reason in failed_models:
        print(f"   - {model}: {reason}")

print(f"\nTotal models trained: {len(successful_models)}/{len(DATASETS)}")
