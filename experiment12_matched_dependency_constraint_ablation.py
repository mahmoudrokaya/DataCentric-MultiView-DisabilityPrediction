from pathlib import Path
import json
import time
import warnings
import hashlib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score, precision_score,
    recall_score, roc_auc_score, average_precision_score,
    cohen_kappa_score, mean_absolute_error
)
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression

warnings.filterwarnings("ignore")

try:
    from catboost import CatBoostClassifier
except Exception:
    CatBoostClassifier = None


# =============================================================================
# EXPERIMENT 12 — MATCHED DEPENDENCY-CONSTRAINT ABLATION
# =============================================================================
#
# Scientific question:
# Does the dependency-constrained structural construction add predictive value
# when the data, split, preprocessing, K/L representation size, learners, OOF
# folds, meta-learner, threshold and metrics are held fixed?
#
# IMPORTANT:
# This script intentionally reproduces Experiment 6's modeling functions and
# settings. The only experimental factor is structural view construction.
# =============================================================================

EVIDENCE_ROOT = Path(
    r"D:\47\472\New-Papers\A Data-Centric Structural Learning Framework -Diagnostics-Rejected\Final\Evidence"
)

EXP_NAME = "experiment12_matched_dependency_constraint_ablation"
OUT_DIR = EVIDENCE_ROOT / "Results" / "exp12_matched_dependency_constraint_ablation"
CODE_DIR = EVIDENCE_ROOT / "Codes" / "exp12_matched_dependency_constraint_ablation"
LOG_DIR = OUT_DIR / "logs"

for d in [OUT_DIR, CODE_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TEST_SIZE = 0.20
OOF_SPLITS = 5
USE_GPU = True

# These MUST match Experiment 6.
BEST_K = 6
BEST_L = 3
STATS = ["mean", "std", "min", "max"]

# Random/shuffled controls are repeated because one random partition can be lucky.
N_RANDOM_REPEATS = 10

TARGET_STAGE = "disability_stage"
TARGET_BINARY = "disability_binary_merge01_vs_23"

ID_COLUMNS = [
    "SEQN", "seqn", "cycle", "Cycle", "id", "ID",
    "index", "Index", "row_id", "original_index"
]

TARGET_SIDE_COLUMNS = [
    "disability_stage",
    "disability_binary_merge01_vs_23",
]

FUNCTIONAL_TARGET_PROXIMAL_COLUMNS = [
    "mobility_difficulty", "adl_difficulty", "upper_limb_difficulty",
    "walking_difficulty", "walk_difficulty", "selfcare_difficulty",
    "self_care_difficulty", "dressing_difficulty", "bathing_difficulty",
    "toilet_difficulty", "transfer_difficulty", "functional_limitation",
]

META_COLUMNS = [
    "is_synthetic", "synthetic", "source", "row_type", "origin",
    "data_source", "synthetic_source_index", "synthetic_neighbor_index",
]

STRUCTURAL_GROUPS = {
    "body_size": ["BMI", "height_cm", "weight_kg"],
    "blood_pressure": ["mean_SBP", "mean_DBP"],
    "lipids": ["total_cholesterol", "HDL", "triglycerides", "LDL"],
    "glycemic": ["HbA1c", "fasting_glucose"],
    "activity": ["MET_min_week", "meets_activity_guideline", "sedentary_minutes_day"],
    "mental_health": ["PHQ9_score", "PHQ9_category"],
    "medication": ["medication_count", "polypharmacy"],
    "arthritis": ["arthritis", "arthritis_type"],
    "demographic": ["age", "sex", "race_ethnicity", "education", "income_poverty_ratio"],
}


def log(msg):
    print(msg)
    with open(LOG_DIR / "experiment12_log.txt", "a", encoding="utf-8") as f:
        f.write(str(msg) + "\n")


def find_unique_file(filename):
    """Find an evidence file without assuming collector subfolder details."""
    matches = [
        p for p in EVIDENCE_ROOT.rglob(filename)
        if EXP_NAME not in str(p)
    ]
    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename!r} anywhere under:\n{EVIDENCE_ROOT}"
        )
    # Prefer Data tree, then shortest path.
    matches.sort(key=lambda p: (0 if "Data" in p.parts else 1, len(p.parts), str(p)))
    if len(matches) > 1:
        log("Multiple candidate data files found:")
        for m in matches:
            log(f"  {m}")
        log(f"Using: {matches[0]}")
    return matches[0]


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def create_binary_target(df):
    if TARGET_STAGE not in df.columns:
        raise ValueError(f"Missing required column: {TARGET_STAGE}")

    stage = pd.to_numeric(df[TARGET_STAGE], errors="coerce")
    df[TARGET_BINARY] = stage.map({0: 0, 1: 0, 2: 1, 3: 1})

    if df[TARGET_BINARY].isna().any():
        bad = df.loc[df[TARGET_BINARY].isna(), TARGET_STAGE].dropna().unique()
        raise ValueError(f"Unexpected disability_stage values: {bad}")

    df[TARGET_BINARY] = df[TARGET_BINARY].astype(int)
    return df


def governance_features(df):
    # Exact Experiment 6 behavior.
    drop_cols = []
    drop_cols += [c for c in ID_COLUMNS if c in df.columns]
    drop_cols += [c for c in TARGET_SIDE_COLUMNS if c in df.columns]
    drop_cols += [c for c in FUNCTIONAL_TARGET_PROXIMAL_COLUMNS if c in df.columns]
    drop_cols += [c for c in META_COLUMNS if c in df.columns]
    drop_cols = sorted(set(drop_cols))

    X = df.drop(columns=drop_cols, errors="ignore").copy()
    encoders = {}

    for col in X.columns:
        if not pd.api.types.is_numeric_dtype(X[col]):
            le = LabelEncoder()
            X[col] = X[col].astype(str).fillna("Unknown")
            X[col] = le.fit_transform(X[col])
            encoders[col] = le

    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    return X, drop_cols, encoders


def build_relation_graph(features, X):
    # Exact Experiment 6 dependency rule.
    feature_set = set(features)
    forbidden_pairs = set()

    for _, cols in STRUCTURAL_GROUPS.items():
        present = [c for c in cols if c in feature_set]
        for i in range(len(present)):
            for j in range(i + 1, len(present)):
                forbidden_pairs.add(tuple(sorted((present[i], present[j]))))

    corr = X[features].corr(numeric_only=True).abs()
    for i, a in enumerate(features):
        for b in features[i + 1:]:
            try:
                if corr.loc[a, b] >= 0.75:
                    forbidden_pairs.add(tuple(sorted((a, b))))
            except Exception:
                pass

    return forbidden_pairs


def is_valid_addition(candidate, current_seq, forbidden_pairs):
    return all(
        tuple(sorted((candidate, existing))) not in forbidden_pairs
        for existing in current_seq
    )


def count_violations(sequences, forbidden_pairs):
    n = 0
    for seq in sequences:
        for i in range(len(seq)):
            for j in range(i + 1, len(seq)):
                if tuple(sorted((seq[i], seq[j]))) in forbidden_pairs:
                    n += 1
    return n


def generate_random_sequences(features, k, length, seed):
    rng = np.random.default_rng(seed)
    return [
        rng.choice(features, size=length, replace=False).tolist()
        for _ in range(k)
    ]


def generate_constraint_valid_random_sequences(features, k, length, forbidden_pairs, seed):
    """
    Random control that obeys exactly the same forbidden-pair rule but does not
    use graph information to guide candidate ordering beyond rejection.
    """
    rng = np.random.default_rng(seed)
    sequences = []

    for _ in range(k):
        success = False
        for _attempt in range(10000):
            seq = rng.choice(features, size=length, replace=False).tolist()
            if count_violations([seq], forbidden_pairs) == 0:
                sequences.append(seq)
                success = True
                break
        if not success:
            raise RuntimeError("Could not generate a constraint-valid random sequence.")
    return sequences


def generate_graph_sequences(features, k, length, forbidden_pairs, seed):
    # Exact Experiment 6 graph sequence generator.
    rng = np.random.default_rng(seed)
    sequences = []

    for _ in range(k):
        shuffled = list(features)
        rng.shuffle(shuffled)

        seq = []
        for candidate in shuffled:
            if len(seq) >= length:
                break
            if is_valid_addition(candidate, seq, forbidden_pairs):
                seq.append(candidate)

        if len(seq) < length:
            for candidate in shuffled:
                if len(seq) >= length:
                    break
                if candidate not in seq:
                    seq.append(candidate)

        sequences.append(seq)

    return sequences


def generate_semantic_sequences(features, k, length, seed):
    """
    Domain-coherent control with exactly K sequences and L features/sequence.
    Domains come from the same STRUCTURAL_GROUPS dictionary already present in
    Experiment 6. Small domains are deterministically completed with unused
    eligible features, while preserving fixed K and L.
    """
    rng = np.random.default_rng(seed)
    feature_set = set(features)

    # Prefer domains with the greatest number of available predictors.
    domains = []
    for name, cols in STRUCTURAL_GROUPS.items():
        present = [c for c in cols if c in feature_set]
        if present:
            domains.append((name, present))
    domains.sort(key=lambda x: (-len(x[1]), x[0]))

    selected = domains[:k]
    sequences = []
    usage = {f: 0 for f in features}

    for _, present in selected:
        seq = list(present[:length])
        for f in seq:
            usage[f] += 1

        if len(seq) < length:
            candidates = [f for f in features if f not in seq]
            candidates.sort(key=lambda f: (usage[f], f))
            # Randomize only within equal-usage blocks for deterministic seed-based tie breaking.
            min_usage = min(usage[f] for f in candidates)
            tied = [f for f in candidates if usage[f] == min_usage]
            rng.shuffle(tied)
            rest = tied + [f for f in candidates if f not in tied]
            for f in rest:
                if len(seq) >= length:
                    break
                seq.append(f)
                usage[f] += 1

        sequences.append(seq)

    if len(sequences) != k or any(len(s) != length for s in sequences):
        raise RuntimeError("Semantic construction failed to preserve matched K/L.")
    return sequences


def shuffle_forbidden_pairs(features, forbidden_pairs, seed):
    """
    Negative control: keep the number of forbidden edges fixed while replacing
    their identities with random feature pairs.
    """
    rng = np.random.default_rng(seed)
    all_pairs = [
        tuple(sorted((features[i], features[j])))
        for i in range(len(features))
        for j in range(i + 1, len(features))
    ]
    n_edges = len(forbidden_pairs)
    idx = rng.choice(len(all_pairs), size=n_edges, replace=False)
    return {all_pairs[i] for i in idx}


def sequence_representation(X, sequences, stats=STATS):
    # Exact Experiment 6 representation.
    reps, names = [], []
    for seq_id, seq in enumerate(sequences, start=1):
        values = X[seq].values
        if "mean" in stats:
            reps.append(values.mean(axis=1)); names.append(f"seq{seq_id}_mean")
        if "std" in stats:
            reps.append(values.std(axis=1)); names.append(f"seq{seq_id}_std")
        if "min" in stats:
            reps.append(values.min(axis=1)); names.append(f"seq{seq_id}_min")
        if "max" in stats:
            reps.append(values.max(axis=1)); names.append(f"seq{seq_id}_max")
    return pd.DataFrame(np.vstack(reps).T, columns=names)


def build_flat_model():
    # Exact Experiment 6 model.
    if CatBoostClassifier is not None:
        args = {
            "loss_function": "Logloss",
            "eval_metric": "AUC",
            "random_seed": RANDOM_STATE,
            "verbose": False,
            "auto_class_weights": "Balanced",
            "allow_writing_files": False,
        }
        if USE_GPU:
            args["task_type"] = "GPU"
            args["devices"] = "0"
        return Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", CatBoostClassifier(**args)),
        ])

    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


def build_structural_model():
    # Exact Experiment 6 model.
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("clf", ExtraTreesClassifier(
            n_estimators=700,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])


def build_meta_model():
    # Exact Experiment 6 meta-learner.
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            max_iter=5000, class_weight="balanced", random_state=RANDOM_STATE
        )),
    ])


def metrics(y_true, prob):
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "macro_f1": f1_score(y_true, pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, pred, average="weighted", zero_division=0),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "kappa": cohen_kappa_score(y_true, pred),
        "mae": mean_absolute_error(y_true, pred),
    }


def get_oof_probs(model_builder, X_train, y_train, X_test, label):
    # Exact Experiment 6 fold definition.
    skf = StratifiedKFold(
        n_splits=OOF_SPLITS, shuffle=True, random_state=RANDOM_STATE
    )
    oof = np.zeros(len(y_train), dtype=float)
    test_folds = []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_train), start=1):
        log(f"OOF {label}, fold {fold}/{OOF_SPLITS}")
        model = model_builder()
        model.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        oof[va_idx] = model.predict_proba(X_train.iloc[va_idx])[:, 1]
        test_folds.append(model.predict_proba(X_test)[:, 1])

    final_model = model_builder()
    final_model.fit(X_train, y_train)
    test_final = final_model.predict_proba(X_test)[:, 1]
    test_oof_avg = np.vstack(test_folds).mean(axis=0)
    return oof, test_final, test_oof_avg, final_model


def make_pair_meta(flat_prob, structural_prob):
    # Exact Experiment 6 OOF meta-features.
    return pd.DataFrame({
        "flat_prob": flat_prob,
        "graph_prob": structural_prob,
        "prob_difference": flat_prob - structural_prob,
        "prob_mean": (flat_prob + structural_prob) / 2.0,
        "prob_product": flat_prob * structural_prob,
    })


def evaluate_flat_oof_control(flat_oof, flat_test, y_train, y_test):
    """
    Plain OOF control absent from Experiment 6.
    Same OOF flat predictions + same logistic meta-learner, but no structural branch.
    """
    meta_train = pd.DataFrame({
        "flat_prob": flat_oof,
        "flat_prob_sq": flat_oof ** 2,
    })
    meta_test = pd.DataFrame({
        "flat_prob": flat_test,
        "flat_prob_sq": flat_test ** 2,
    })
    meta = build_meta_model()
    meta.fit(meta_train, y_train)
    prob = meta.predict_proba(meta_test)[:, 1]
    return prob


def evaluate_structural_oof(
    label, sequences, X, train_idx, test_idx, y_train, y_test,
    flat_oof, flat_test, true_forbidden_pairs, replicate
):
    X_seq = sequence_representation(X, sequences, STATS)
    X_train_seq = X_seq.iloc[train_idx].reset_index(drop=True)
    X_test_seq = X_seq.iloc[test_idx].reset_index(drop=True)

    start = time.time()
    struct_oof, struct_test, _, _ = get_oof_probs(
        build_structural_model,
        X_train_seq, y_train, X_test_seq, label
    )

    meta_train = make_pair_meta(flat_oof, struct_oof)
    meta_test = make_pair_meta(flat_test, struct_test)

    meta = build_meta_model()
    meta.fit(meta_train, y_train)
    prob = meta.predict_proba(meta_test)[:, 1]
    runtime = time.time() - start

    row = {
        "configuration": label,
        "replicate": replicate,
        "K": BEST_K,
        "L": BEST_L,
        "n_representation_features": X_seq.shape[1],
        "true_constraint_violations": count_violations(sequences, true_forbidden_pairs),
        "runtime_seconds": runtime,
        **metrics(y_test, prob),
    }
    return row, prob


def summarize_repeated_results(results):
    df = pd.DataFrame(results)
    metrics_cols = [
        "macro_f1", "weighted_f1", "balanced_accuracy",
        "roc_auc", "pr_auc", "kappa", "mae"
    ]
    rows = []
    for config, g in df.groupby("configuration", sort=False):
        row = {
            "configuration": config,
            "n_runs": len(g),
            "mean_true_constraint_violations": g["true_constraint_violations"].mean(),
        }
        for m in metrics_cols:
            row[f"{m}_mean"] = g[m].mean()
            row[f"{m}_sd"] = g[m].std(ddof=1) if len(g) > 1 else 0.0
            row[f"{m}_min"] = g[m].min()
            row[f"{m}_max"] = g[m].max()
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    log_file = LOG_DIR / "experiment12_log.txt"
    if log_file.exists():
        log_file.unlink()

    log("=" * 100)
    log("EXPERIMENT 12: MATCHED DEPENDENCY-CONSTRAINT ABLATION")
    log("=" * 100)

    data_file = find_unique_file("NHANES_pooled_filled_corrected_blue.xlsx")
    log(f"DATA_FILE = {data_file}")
    log(f"SHA256 = {sha256(data_file)}")

    df = pd.read_excel(data_file)
    df = create_binary_target(df)

    X, dropped_cols, _ = governance_features(df)
    y = df[TARGET_BINARY].values.astype(int)
    features = list(X.columns)

    forbidden_pairs = build_relation_graph(features, X)

    train_idx, test_idx = train_test_split(
        np.arange(len(y)),
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE
    )
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    y_train = y[train_idx]
    y_test = y[test_idx]

    # -------------------------------------------------------------------------
    # Shared flat OOF branch — computed ONCE and reused for every configuration.
    # -------------------------------------------------------------------------
    log("\nComputing shared flat OOF branch...")
    flat_oof, flat_test, _, _ = get_oof_probs(
        build_flat_model, X_train, y_train, X_test, "flat_shared"
    )

    results = []
    predictions = pd.DataFrame({"y_true": y_test})

    # A: plain OOF flat control
    log("\nE12-A: Flat OOF control")
    start = time.time()
    prob_flat_oof = evaluate_flat_oof_control(
        flat_oof, flat_test, y_train, y_test
    )
    row = {
        "configuration": "A_flat_oof_control",
        "replicate": 0,
        "K": 0, "L": 0,
        "n_representation_features": 0,
        "true_constraint_violations": 0,
        "runtime_seconds": time.time() - start,
        **metrics(y_test, prob_flat_oof),
    }
    results.append(row)
    predictions["A_flat_oof_control"] = prob_flat_oof

    # E: actual Experiment-6 graph construction (primary proposed method)
    graph_sequences = generate_graph_sequences(
        features, BEST_K, BEST_L, forbidden_pairs,
        RANDOM_STATE + BEST_K + BEST_L
    )
    log("\nE12-E: Dependency-constrained graph views")
    row, prob = evaluate_structural_oof(
        "E_dependency_graph", graph_sequences, X,
        train_idx, test_idx, y_train, y_test,
        flat_oof, flat_test, forbidden_pairs, 0
    )
    results.append(row)
    predictions["E_dependency_graph"] = prob

    # D: semantic/domain control
    semantic_sequences = generate_semantic_sequences(
        features, BEST_K, BEST_L, RANDOM_STATE
    )
    log("\nE12-D: Semantic/domain views")
    row, prob = evaluate_structural_oof(
        "D_semantic_domain", semantic_sequences, X,
        train_idx, test_idx, y_train, y_test,
        flat_oof, flat_test, forbidden_pairs, 0
    )
    results.append(row)
    predictions["D_semantic_domain"] = prob

    # B/C/F: repeated stochastic controls
    sequence_manifest = {
        "E_dependency_graph": graph_sequences,
        "D_semantic_domain": semantic_sequences,
    }

    for r in range(N_RANDOM_REPEATS):
        seed = RANDOM_STATE + 1000 + r

        log(f"\nE12-B replicate {r+1}/{N_RANDOM_REPEATS}: Unconstrained random views")
        seq = generate_random_sequences(features, BEST_K, BEST_L, seed)
        row, prob = evaluate_structural_oof(
            "B_random_unconstrained", seq, X,
            train_idx, test_idx, y_train, y_test,
            flat_oof, flat_test, forbidden_pairs, r + 1
        )
        results.append(row)
        predictions[f"B_random_unconstrained_r{r+1}"] = prob
        sequence_manifest[f"B_random_unconstrained_r{r+1}"] = seq

        log(f"\nE12-C replicate {r+1}/{N_RANDOM_REPEATS}: Constraint-valid random views")
        seq = generate_constraint_valid_random_sequences(
            features, BEST_K, BEST_L, forbidden_pairs, seed
        )
        row, prob = evaluate_structural_oof(
            "C_random_constraint_valid", seq, X,
            train_idx, test_idx, y_train, y_test,
            flat_oof, flat_test, forbidden_pairs, r + 1
        )
        results.append(row)
        predictions[f"C_random_constraint_valid_r{r+1}"] = prob
        sequence_manifest[f"C_random_constraint_valid_r{r+1}"] = seq

        log(f"\nE12-F replicate {r+1}/{N_RANDOM_REPEATS}: Shuffled dependency graph")
        shuffled_pairs = shuffle_forbidden_pairs(features, forbidden_pairs, seed)
        seq = generate_graph_sequences(
            features, BEST_K, BEST_L, shuffled_pairs,
            seed + BEST_K + BEST_L
        )
        row, prob = evaluate_structural_oof(
            "F_shuffled_graph", seq, X,
            train_idx, test_idx, y_train, y_test,
            flat_oof, flat_test, forbidden_pairs, r + 1
        )
        results.append(row)
        predictions[f"F_shuffled_graph_r{r+1}"] = prob
        sequence_manifest[f"F_shuffled_graph_r{r+1}"] = seq

    results_df = pd.DataFrame(results)
    summary_df = summarize_repeated_results(results)

    # Compare proposed graph result with stochastic-control distributions.
    graph_row = results_df[
        results_df["configuration"] == "E_dependency_graph"
    ].iloc[0]

    comparison_rows = []
    for control in [
        "B_random_unconstrained",
        "C_random_constraint_valid",
        "F_shuffled_graph",
    ]:
        g = results_df[results_df["configuration"] == control]
        for metric in ["macro_f1", "roc_auc", "pr_auc", "balanced_accuracy"]:
            vals = g[metric].to_numpy()
            proposed = float(graph_row[metric])
            comparison_rows.append({
                "control": control,
                "metric": metric,
                "proposed_graph_value": proposed,
                "control_mean": float(vals.mean()),
                "control_sd": float(vals.std(ddof=1)),
                "proposed_minus_control_mean": proposed - float(vals.mean()),
                "fraction_control_runs_ge_proposed": float(np.mean(vals >= proposed)),
            })
    comparison_df = pd.DataFrame(comparison_rows)

    # Reproduction guard against Experiment 6.
    # The graph OOF result should be extremely close to the reported Experiment 6
    # value if the collector retained the same source data and software behavior.
    reproduction = {
        "experiment6_reported_macro_f1": 0.720911,
        "experiment6_reported_roc_auc": 0.789515,
        "experiment12_graph_macro_f1": float(graph_row["macro_f1"]),
        "experiment12_graph_roc_auc": float(graph_row["roc_auc"]),
        "abs_macro_f1_difference": abs(float(graph_row["macro_f1"]) - 0.720911),
        "abs_roc_auc_difference": abs(float(graph_row["roc_auc"]) - 0.789515),
    }
    reproduction["passes_1e-4_tolerance"] = (
        reproduction["abs_macro_f1_difference"] <= 1e-4
        and reproduction["abs_roc_auc_difference"] <= 1e-4
    )

    setup = {
        "experiment": EXP_NAME,
        "scientific_question":
            "Does dependency-constrained view construction add value under an otherwise identical OOF pipeline?",
        "data_file": str(data_file),
        "data_sha256": sha256(data_file),
        "n_rows": int(len(df)),
        "n_features": int(X.shape[1]),
        "features": features,
        "dropped_columns": dropped_cols,
        "target": TARGET_BINARY,
        "binary_rule": "disability_stage 0/1 -> 0; 2/3 -> 1",
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "oof_splits": OOF_SPLITS,
        "K": BEST_K,
        "L": BEST_L,
        "stats": STATS,
        "n_random_repeats": N_RANDOM_REPEATS,
        "n_forbidden_pairs": len(forbidden_pairs),
        "forbidden_pairs": sorted([list(x) for x in forbidden_pairs]),
        "train_n": int(len(train_idx)),
        "test_n": int(len(test_idx)),
        "reproduction_guard": reproduction,
    }

    results_df.to_csv(
        OUT_DIR / "experiment12_all_runs.csv",
        index=False, encoding="utf-8-sig"
    )
    summary_df.to_csv(
        OUT_DIR / "experiment12_summary.csv",
        index=False, encoding="utf-8-sig"
    )
    comparison_df.to_csv(
        OUT_DIR / "experiment12_graph_vs_controls.csv",
        index=False, encoding="utf-8-sig"
    )
    predictions.to_csv(
        OUT_DIR / "experiment12_test_predictions.csv",
        index=False, encoding="utf-8-sig"
    )

    with open(OUT_DIR / "experiment12_setup_summary.json", "w", encoding="utf-8") as f:
        json.dump(setup, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "experiment12_sequence_manifest.json", "w", encoding="utf-8") as f:
        json.dump(sequence_manifest, f, indent=2, ensure_ascii=False)

    with open(OUT_DIR / "experiment12_reproduction_guard.json", "w", encoding="utf-8") as f:
        json.dump(reproduction, f, indent=2, ensure_ascii=False)

    log("\n" + "=" * 100)
    log("EXPERIMENT 12 COMPLETED")
    log("=" * 100)
    log("\nReproduction guard:")
    log(json.dumps(reproduction, indent=2))
    log("\nSummary:")
    log(summary_df.to_string(index=False))
    log("\nGraph vs stochastic controls:")
    log(comparison_df.to_string(index=False))
    log(f"\nResults saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
