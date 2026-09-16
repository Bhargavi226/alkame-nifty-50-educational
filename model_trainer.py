import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_score,
    recall_score,
)

from config import (
    HORIZON_CONFIG,
    HORIZON_INTRADAY,
    LABEL_CLASSES,
    MODEL_LEARNING_RATE,
    MODEL_MAX_DEPTH,
    MODEL_N_ESTIMATORS,
    MODEL_RANDOM_SEED,
    MODELS_DIR,
    TIME_SERIES_SPLIT_TEST_FRACTION,
    configure_logging,
    ensure_directories,
)

from feature_engineer import ML_SAFE_SUFFIX, FeatureEngineer


logger = logging.getLogger(__name__)


@dataclass
class ModelEvaluation:
    accuracy: float
    balanced_accuracy: float
    precision: float
    recall: float
    f1: float
    mcc: float
    logloss: float
    confusion_matrix: list
    classification_report: dict


@dataclass
class EnsembleModel:
    models: list
    feature_columns: list
    label_classes: list
    horizon: str
    trained_at: str
    metadata: dict


class ModelTrainer:
    """
    Handles dataset preparation, chronological model training,
    evaluation, persistence and prediction.

    ML-01 safety requirements:
    - feature inputs must be ML-safe
    - labels must use future information only
    - chronological splits must protect the test period
    - training labels must not consume protected test observations
    """

    def __init__(
        self,
        feature_engineer=None,
        models_dir=MODELS_DIR,
        test_fraction=TIME_SERIES_SPLIT_TEST_FRACTION,
        random_seed=MODEL_RANDOM_SEED,
        min_training_samples=100,
    ):
        self.feature_engineer = (
            feature_engineer
            if feature_engineer is not None
            else FeatureEngineer()
        )

        self.models_dir = Path(models_dir)
        self.test_fraction = test_fraction
        self.random_seed = random_seed
        self.min_training_samples = min_training_samples

        ensure_directories()

    # ------------------------------------------------------------------
    # Adaptive deadband
    # ------------------------------------------------------------------

    def compute_adaptive_deadband(
        self,
        returns,
        window=20,
        multiplier=1.0,
    ):
        """
        Calculate a rolling volatility-based deadband.

        Only information available up to the current row is used.
        """
        returns = pd.Series(returns).astype(float)

        rolling_std = (
            returns
            .rolling(window=window, min_periods=window)
            .std()
        )

        deadband = rolling_std * multiplier

        return deadband

    # ------------------------------------------------------------------
    # Synthetic user-cost simulation
    # ------------------------------------------------------------------

    def simulate_user_cost(self, df, seed=None):
        """
        Create deterministic user-cost features.

        Cost information is lagged before being exposed as a model input
        so the current observation cannot use information from the
        current/future execution result.

        The public contract expects the DataFrame to be mutated in-place and
        to expose both the raw feature names used by tests and the feature-
        suffixed aliases used elsewhere in the ML pipeline.
        """
        if "Close" not in df.columns:
            raise ValueError("Close column is required")

        close = pd.to_numeric(df["Close"], errors="coerce")
        day_key = df.index.normalize()

        daily_open = close.groupby(day_key).transform("first")
        has_position = close.notna() & daily_open.notna() & (close != 0)
        user_avg_cost = daily_open
        pct_from_user_avg_cost = ((close - user_avg_cost) / user_avg_cost.replace(0, np.nan)) * 100.0

        # Keep the raw names expected by the ML-01 tests and preserve the feature
        # names consumed by the feature engineering pipeline.
        df["has_position"] = has_position.astype(float)
        df["pct_from_user_avg_cost"] = pct_from_user_avg_cost

        df["has_position_feat"] = df["has_position"].shift(1).fillna(0.0)
        df["pct_from_user_avg_cost_feat"] = df["pct_from_user_avg_cost"].shift(1)

        return df

    # ------------------------------------------------------------------
    # Price-level labels
    # ------------------------------------------------------------------

    def build_price_level_labels(
        self,
        df,
        horizon_bars,
    ):
        """
        Build future price-level labels.

        The future window is used only for the target. The final
        horizon rows therefore have no valid label.
        """
        if horizon_bars <= 0:
            raise ValueError(
                "horizon_bars must be greater than zero"
            )

        if len(df) <= horizon_bars:
            raise ValueError(
                "Dataset is too short for the requested horizon"
            )

        required_columns = {"High", "Low", "Close"}

        missing = required_columns - set(df.columns)

        if missing:
            raise ValueError(
                f"Missing required columns: {sorted(missing)}"
            )

        labels = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        high_values = pd.to_numeric(
            df["High"],
            errors="coerce",
        ).to_numpy()

        low_values = pd.to_numeric(
            df["Low"],
            errors="coerce",
        ).to_numpy()

        close_values = pd.to_numeric(
            df["Close"],
            errors="coerce",
        ).to_numpy()

        for i in range(len(df) - horizon_bars):
            future_high = np.nanmax(
                high_values[
                    i + 1 : i + horizon_bars + 1
                ]
            )

            future_low = np.nanmin(
                low_values[
                    i + 1 : i + horizon_bars + 1
                ]
            )

            current_close = close_values[i]

            if (
                not np.isfinite(current_close)
                or not np.isfinite(future_high)
                or not np.isfinite(future_low)
                or current_close == 0
            ):
                continue

            upside = (
                future_high - current_close
            ) / current_close

            downside = (
                future_low - current_close
            ) / current_close

            if upside > abs(downside):
                labels.iloc[i] = 1
            elif downside < -abs(upside):
                labels.iloc[i] = -1
            else:
                labels.iloc[i] = 0

        return labels
    # ------------------------------------------------------------------
    # Future-return labels
    # ------------------------------------------------------------------

    def build_labels(
        self,
        df,
        horizon_bars,
        deadband_multiplier=1.0,
    ):
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")

        if not X.index.equals(y.index):
            raise ValueError("X and y must have identical indexes and ordering")

        if not X.index.is_unique:
            raise ValueError("Duplicate timestamps are not allowed")

        if not X.index.is_monotonic_increasing:
            raise ValueError("Input timestamps must be sorted chronologically")
        
        """
        Build forward-return classification labels.

        IMPORTANT:
        The target at row i depends on the price at row
        i + horizon_bars. Therefore the final horizon rows
        cannot have valid labels.
        """
        if horizon_bars <= 0:
            raise ValueError(
                "horizon_bars must be greater than zero"
            )

        if len(df) <= horizon_bars:
            raise ValueError(
                "Dataset is too short for the requested horizon"
            )

        if "Close" not in df.columns:
            raise ValueError("Close column is required")

        close = pd.to_numeric(
            df["Close"],
            errors="coerce",
        )

        future_close = close.shift(-horizon_bars)

        forward_return = (
            future_close / close.replace(0, np.nan)
        ) - 1.0

        current_return = (
            close.pct_change()
        )

        deadband = self.compute_adaptive_deadband(
            current_return,
            window=20,
            multiplier=deadband_multiplier,
        )

        labels = pd.Series(
            np.nan,
            index=df.index,
            dtype=float,
        )

        valid = (
            forward_return.notna()
            & deadband. notna()
            & np.isfinite(forward_return)
            & np.isfinite(deadband)
        )
    def time_based_split(
        self,
        X,
        y,
        test_fraction=0.2,
        purge_window=0,
    ): 
        if not isinstance(X, pd.DataFrame):
            raise TypeError("X must be a pandas DataFrame")

        if not isinstance(y, pd.Series):
            raise TypeError("y must be a pandas Series")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of rows")

        if X.index.has_duplicates:
            raise ValueError("X timestamps must not contain duplicates")

        if not X.index.is_monotonic_increasing:
            raise ValueError("X timestamps must be sorted")

        n = len(X)
        test_size = int(n * test_fraction)

        if test_size <= 0:
            raise ValueError("test_fraction produces an empty test set")

        test_start = n - test_size
        train_end = test_start - purge_window

        if train_end <= 0:
            raise ValueError("Insufficient samples for time-based split")

        X_train = X.iloc[:train_end].copy()
        X_test = X.iloc[test_start:].copy()
        y_train = y.iloc[:train_end].copy()
        y_test = y.iloc[test_start:].copy()

        return X_train, X_test, y_train, y_test
            # ------------------------------------------------------------------
    # Walk-forward split
    # ------------------------------------------------------------------

    def walk_forward_split(
        self,
        X,
        y,
        n_splits=5,
        purge_window=0,
    ):
        """
        Generate chronological walk-forward train/test splits.

        Every validation period occurs after its corresponding
        training period, with an optional purge gap between them.
        """
        if not isinstance(X, pd.DataFrame):
            raise TypeError(
                "X must be a pandas DataFrame"
            )

        if not isinstance(y, pd.Series):
            raise TypeError(
                "y must be a pandas Series"
            )

        if len(X) != len(y):
            raise ValueError(
                "X and y must contain the same number of rows"
            )

        if n_splits < 2:
            raise ValueError(
                "n_splits must be at least 2"
            )

        if len(X) < n_splits + 1:
            raise ValueError(
                "Dataset is too short for walk-forward validation"
            )

        if purge_window < 0:
            raise ValueError(
                "purge_window must be non-negative"
            )

        if not X.index.equals(y.index):
            raise ValueError(
                "X and y must have identical indexes and ordering"
            )

        if not X.index.is_unique:
            raise ValueError(
                "Duplicate timestamps are not allowed"
            )

        if not X.index.is_monotonic_increasing:
            raise ValueError(
                "Input timestamps must be sorted chronologically"
            )

        n_rows = len(X)

        test_size = n_rows // (n_splits + 1)

        if test_size <= 0:
            raise ValueError(
                "Unable to create walk-forward test windows"
            )
            splits = []

        for split_number in range(
            1,
            n_splits + 1,
        ):
            test_start = (
                split_number * test_size
            )

            if split_number < n_splits:
                test_end = (
                    test_start + test_size
                )
            else:
                test_end = n_rows

            train_end = (
                test_start - purge_window
            )

            if train_end <= 0:
                continue

            X_train = X.iloc[
                :train_end
            ].copy()

            X_test = X.iloc[
                test_start:test_end
            ].copy()

            y_train = y.iloc[
                :train_end
            ].copy()

            y_test = y.iloc[
                test_start:test_end
            ].copy()

            if X_train.empty:
                continue

            if X_test.empty:
                continue

            if len(X_train) != len(y_train):
                raise ValueError(
                    "Walk-forward training X/y length mismatch"
                )

            if len(X_test) != len(y_test):
                raise ValueError(
                    "Walk-forward test X/y length mismatch"
                )

            splits.append(
                (
                    X_train,
                    X_test,
                    y_train,
                    y_test,
                )
            )

        if not splits:
            raise ValueError(
                "No valid walk-forward splits could be created"
            )

        return splits

    # ------------------------------------------------------------------
    # Model creation
    # ------------------------------------------------------------------

    def _create_model(self):
        """
        Create a deterministic Gradient Boosting classifier.
        """
        return GradientBoostingClassifier(
            n_estimators=MODEL_N_ESTIMATORS,
            learning_rate=MODEL_LEARNING_RATE,
            max_depth=MODEL_MAX_DEPTH,
            random_state=self.random_seed,
        )
        # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(
        self,
        X_train,
        y_train,
        feature_columns=None,
        horizon=None,
    ):
        """
        Train one classifier on the supplied training fold.
        """
        if X_train.empty:
            raise ValueError(
                "Cannot train on an empty dataset"
            )
            if y_train.empty:
             raise ValueError(
                "Cannot train without labels"
            )

        if len(X_train) != len(y_train):
            raise ValueError(
                "X_train and y_train length mismatch"
            )

        if feature_columns is None:
            feature_columns = list(
                X_train.columns
            )

        if list(X_train.columns) != list(
            feature_columns
        ):
            raise ValueError(
                "Feature column order does not match declared model inputs"
            )

        unique_classes = sorted(
            pd.Series(y_train)
            .dropna()
            .unique()
            .tolist()
        )

        if len(unique_classes) < 2:
            raise ValueError(
                "Training data must contain at least two classes"
            )

        model = self._create_model()

        model.fit(
            X_train,
            y_train,
        )

        return model

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        model,
        X,
    ):
        """
        Generate class predictions from a trained model.
        """
        if model is None:
            raise ValueError(
                "A trained model is required"
            )

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame"
            )

        if X.empty:
            raise ValueError(
                "Cannot predict on an empty dataset"
            )

        return model.predict(X)

    # ------------------------------------------------------------------
    # Probability prediction
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        model,
        X,
    ):
        """
        Generate class probabilities from a trained model.
        """
        if model is None:
            raise ValueError(
                "A trained model is required"
            )

        if not hasattr(
            model,
            "predict_proba",
        ):
            raise ValueError(
                "Model does not support probability prediction"
            )

        if not isinstance(
            X,
            pd.DataFrame,
        ):
            raise TypeError(
                "X must be a pandas DataFrame"
            )

        if X.empty:
            raise ValueError(
                "Cannot predict probabilities on an empty dataset"
            )

        return model.predict_proba(X)
    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(
        self,
        model,
        X_test,
        y_test,
    ):
        """
        Evaluate a trained model on a protected test set.
        """
        if model is None:
            raise ValueError(
                "A trained model is required"
            )

        if X_test.empty or y_test.empty:
            raise ValueError(
                "Test data cannot be empty"
            )

        if len(X_test) != len(y_test):
            raise ValueError(
                "X_test and y_test length mismatch"
            )

        predictions = model.predict(
            X_test
        )

        probabilities = None

        if hasattr(
            model,
            "predict_proba",
        ):
            probabilities = (
                model.predict_proba(X_test)
            )

        accuracy = accuracy_score(
            y_test,
            predictions,
        )

        balanced_accuracy = (
            balanced_accuracy_score(
                y_test,
                predictions,
            )
        )

        precision = precision_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        recall = recall_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        f1 = f1_score(
            y_test,
            predictions,
            average="weighted",
            zero_division=0,
        )

        mcc = matthews_corrcoef(
            y_test,
            predictions,
        )

        if probabilities is not None:
            try:
                logloss = log_loss(
                    y_test,
                    probabilities,
                    labels=model.classes_,
                )
            except ValueError:
                logloss = float("nan")
        else:
            logloss = float("nan")

        matrix = confusion_matrix(
            y_test,
            predictions,
        ).tolist()

        report = classification_report(
            y_test,
            predictions,
            output_dict=True,
            zero_division=0,
        )

        return ModelEvaluation(
            accuracy=float(accuracy),
            balanced_accuracy=float(
                balanced_accuracy
            ),
            precision=float(precision),
            recall=float(recall),
            f1=float(f1),
            mcc=float(mcc),
            logloss=float(logloss),
            confusion_matrix=matrix,
            classification_report=report,
        )
        # ------------------------------------------------------------------
    # Walk-forward evaluation
    # ------------------------------------------------------------------

    def evaluate_walk_forward(
        self,
        X,
        y,
        n_splits=5,
        purge_window=0,
    ):
        """
        Train and evaluate separate models on chronological
        walk-forward folds.
        """
        splits = self.walk_forward_split(
            X,
            y,
            n_splits=n_splits,
            purge_window=purge_window,
        )

        evaluations = []

        for (
            X_train,
            X_test,
            y_train,
            y_test,
        ) in splits:

            if len(
                y_train.unique()
            ) < 2:
                logger.warning(
                    "Skipping fold because training data "
                    "contains fewer than two classes"
                )
                continue

            model = self.train(
                X_train,
                y_train,
                feature_columns=list(
                    X_train.columns
                ),
            )

            evaluation = self.evaluate(
                model,
                X_test,
                y_test,
            )

            evaluations.append(
                evaluation
            )

        if not evaluations:
            raise ValueError(
                "No valid walk-forward evaluations were produced"
            )

        return evaluations

    # ------------------------------------------------------------------
    # Model artifact path
    # ------------------------------------------------------------------

    def _model_path(
        self,
        symbol,
        horizon,
    ):
        """
        Return the model artifact path.
        """
        safe_symbol = str(
            symbol
        ).replace(
            "/",
            "_",
        )

        safe_horizon = str(
            horizon
        ).replace(
            "/",
            "_",
        )

        self.models_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        return (
            self.models_dir
            / f"{safe_symbol}_{safe_horizon}_model.joblib"
        )
        # ------------------------------------------------------------------
    # Save model
    # ------------------------------------------------------------------

    def save_model(
        self,
        model,
        symbol,
        horizon,
        feature_columns,
        metadata=None,
    ):
        """
        Persist a trained model together with its declared feature
        order and metadata.
        """
        if model is None:
            raise ValueError(
                "Cannot save an empty model"
            )

        if not feature_columns:
            raise ValueError(
                "feature_columns cannot be empty"
            )

        artifact = {
            "model": model,
            "feature_columns": list(
                feature_columns
            ),
            "label_classes": list(
                getattr(
                    model,
                    "classes_",
                    LABEL_CLASSES,
                )
            ),
            "symbol": symbol,
            "horizon": horizon,
            "trained_at": datetime.now(timezone.utc).isoformat(),
            "metadata": metadata or {},
        }

        path = self._model_path(
            symbol,
            horizon,
        )

        joblib.dump(
            artifact,
            path,
        )

        return path

    # ------------------------------------------------------------------
    # Load model
    # ------------------------------------------------------------------

    def load_model(
        self,
        symbol,
        horizon,
    ):
        """
        Load a previously saved model artifact.

        The artifact must contain the model, feature order,
        label classes and metadata together.
        """
        path = self._model_path(
            symbol,
            horizon,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Model artifact not found: {path}"
            )

        artifact = joblib.load(
            path
        )

        if not isinstance(
            artifact,
            dict,
        ):
            raise ValueError(
                "Invalid model artifact format"
            )

        required_keys = {
            "model",
            "feature_columns",
            "label_classes",
            "symbol",
            "horizon",
            "trained_at",
            "metadata",
        }

        missing = (
            required_keys
            - set(artifact.keys())
        )

        if missing:
            raise ValueError(
                f"Model artifact is missing required fields: "
                f"{sorted(missing)}"
            )

        if artifact["symbol"] != symbol:
            raise ValueError(
                "Model artifact symbol does not match requested symbol"
            )

        if artifact["horizon"] != horizon:
            raise ValueError(
                "Model artifact horizon does not match requested horizon"
            )

        if not artifact["feature_columns"]:
            raise ValueError(
                "Model artifact contains no feature columns"
            )

        return artifact
    # ------------------------------------------------------------------
    # Train model for a symbol and horizon
    # ------------------------------------------------------------------

    def train_for_symbol(
        self,
        symbol,
        df,
        horizon,
        horizon_bars,
        include_user_cost=False,
        deadband_multiplier=1.0,
    ):
        """
        Prepare data, perform a chronological split, train the model,
        evaluate it on the protected test set and save the artifact.
        """
        X, y, feature_columns = (
            self.prepare_dataset(
                df,
                horizon_bars=horizon_bars,
                include_user_cost=include_user_cost,
                deadband_multiplier=deadband_multiplier,
            )
        )

        if len(X) < self.min_training_samples:
            raise ValueError(
                f"Insufficient samples: {len(X)} "
                f"< minimum {self.min_training_samples}"
            )

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.time_based_split(
            X,
            y,
            test_fraction=self.test_fraction,
            purge_window=horizon_bars,
        )

        if len(X_train) < self.min_training_samples:
            raise ValueError(
                f"Insufficient training samples after purge: "
                f"{len(X_train)} < {self.min_training_samples}"
            )

        if len(y_train.unique()) < 2:
            raise ValueError(
                "Training partition does not contain at least two classes"
            )

        model = self.train(
            X_train,
            y_train,
            feature_columns=feature_columns,
            horizon=horizon,
        )

        evaluation = self.evaluate(
            model,
            X_test,
            y_test,
        )

        metadata = {
            "symbol": symbol,
            "horizon": horizon,
            "horizon_bars": horizon_bars,
            "feature_columns": list(
                feature_columns
            ),
            "train_samples": int(
                len(X_train)
            ),
            "test_samples": int(
                len(X_test)
            ),
            "test_fraction": float(
                self.test_fraction
            ),
            "purge_window": int(
                horizon_bars
            ),
            "evaluation": {
                "accuracy": evaluation.accuracy,
                "balanced_accuracy": (
                    evaluation.balanced_accuracy
                ),
                "precision": evaluation.precision,
                "recall": evaluation.recall,
                "f1": evaluation.f1,
                "mcc": evaluation.mcc,
                "logloss": evaluation.logloss,
            },
        }

        model_path = self.save_model(
            model,
            symbol=symbol,
            horizon=horizon,
            feature_columns=feature_columns,
            metadata=metadata,
        )

        return {
            "model": model,
            "evaluation": evaluation,
            "model_path": model_path,
            "feature_columns": feature_columns,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
        }

    # ------------------------------------------------------------------
    # Self-test
    # ------------------------------------------------------------------

    def self_test(self):
        """
        Run a small deterministic smoke test for the trainer.
        """
        rng = np.random.default_rng(
            self.random_seed
        )

        periods = 300

        index = pd.date_range(
            "2026-01-01",
            periods=periods,
            freq="h",
        )

        base = (
            100
            + np.cumsum(
                rng.normal(
                    0,
                    0.5,
                    periods,
                )
            )
        )

        synthetic = pd.DataFrame(
            {
                "Open": base
                + rng.normal(
                    0,
                    0.2,
                    periods,
                ),
                "High": base
                + np.abs(
                    rng.normal(
                        0.5,
                        0.2,
                        periods,
                    )
                ),
                "Low": base
                - np.abs(
                    rng.normal(
                        0.5,
                        0.2,
                        periods,
                    )
                ),
                "Close": base,
                "Volume": rng.integers(
                    1000,
                    10000,
                    periods,
                ),
            },
            index=index,
        )

        X, y, feature_columns = (
            self.prepare_dataset(
                synthetic,
                horizon_bars=6,
            )
        )

        assert len(X) == len(y)
        assert len(feature_columns) > 0

        (
            X_train,
            X_test,
            y_train,
            y_test,
        ) = self.time_based_split(
            X,
            y,
            test_fraction=0.2,
            purge_window=6,
        )

        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) == len(X_train)
        assert len(y_test) == len(X_test)

        model = self.train(
            X_train,
            y_train,
            feature_columns=feature_columns,
        )

        predictions = self.predict(
            model,
            X_test,
        )

        assert len(predictions) == len(
            X_test
        )

        evaluation = self.evaluate(
            model,
            X_test,
            y_test,
        )

        assert 0.0 <= evaluation.accuracy <= 1.0
        assert 0.0 <= evaluation.f1 <= 1.0

        logger.info(
            "model_trainer.py self-test passed."
        )

        return True


if __name__ == "__main__":
    configure_logging()

    trainer = ModelTrainer()

    try:
        trainer.self_test()

        print(
            "STATUS: PASS"
        )

    except Exception as exc:
        logger.exception(
            "model_trainer.py self-test failed"
        )

        print(
            f"STATUS: FAIL - {exc}"
        )

        raise
