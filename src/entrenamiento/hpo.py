"""Búsqueda de hiperparámetros con Optuna, registrada en MLflow como parent/child runs."""

from time import perf_counter

import mlflow
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import GridSampler, TPESampler
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from entrenamiento.configuracion import (
    FEATURES,
    SEED,
    TAGS_PROYECTO,
    TARGET,
    nombre_familia,
)
from entrenamiento.features import construir_pipeline

optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = {"ridge": 13, "random_forest": 12, "lightgbm": 25}
REJILLA_RIDGE = {"alpha": np.logspace(-1, 5, 13).tolist()}


def evaluar(y_real, y_pred) -> dict[str, float]:
    """Métricas sobre y = log1p(play_count). `rmsle` es la métrica principal (minimizar)."""
    return {
        "rmsle": float(root_mean_squared_error(y_real, y_pred)),
        "r2": float(r2_score(y_real, y_pred)),
        "mae_log": float(mean_absolute_error(y_real, y_pred)),
        "spearman": float(spearmanr(y_real, y_pred).statistic),
    }


def proponer_parametros(trial: optuna.Trial, familia: str) -> dict:
    """Espacio de búsqueda de cada familia (sólo los hiperparámetros que se ajustan)."""
    if familia == "ridge":
        return {"alpha": trial.suggest_float("alpha", 1e-1, 1e5, log=True)}
    if familia == "random_forest":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 50, 200, step=50),
            "max_depth": trial.suggest_int("max_depth", 8, 24),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 2, 100, log=True),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", 0.3, 0.5]),
            "max_samples": trial.suggest_float("max_samples", 0.3, 1.0),
        }
    if familia == "lightgbm":
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.2, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 15, 255, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 20, 300, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
        }
    raise ValueError(f"Familia no reconocida: {familia}")


def completar_parametros(familia: str, ajustados: dict) -> dict:
    """Agrega los parámetros fijos que no forman parte del espacio de búsqueda."""
    fijos = {
        "ridge": {},
        "random_forest": {"random_state": SEED, "n_jobs": -1},
        "lightgbm": {"subsample_freq": 1, "random_state": SEED, "n_jobs": -1, "verbose": -1},
    }
    return {**ajustados, **fijos[familia]}


def crear_estudio(familia: str) -> tuple[optuna.Study, str]:
    """Ridge: rejilla exhaustiva. Random Forest y LightGBM: TPE con semilla fija."""
    if familia == "ridge":
        sampler, nombre = GridSampler(REJILLA_RIDGE, seed=SEED), "optuna-grid"
    else:
        sampler, nombre = TPESampler(seed=SEED, n_startup_trials=5), "optuna-tpe"
    return optuna.create_study(direction="minimize", sampler=sampler), nombre


def ajustar_familia(
    familia: str,
    train: pd.DataFrame,
    validacion: pd.DataFrame,
    n_trials: int | None = None,
) -> dict:
    """HPO de una familia: parent run `hpo-<familia>` con un child run por trial.

    Sólo recibe train y validación: el test no puede entrar a la función objetivo.
    """
    n_trials = n_trials or N_TRIALS[familia]
    estudio, sampler = crear_estudio(familia)

    def objective(trial: optuna.Trial) -> float:
        ajustados = proponer_parametros(trial, familia)
        pipeline = construir_pipeline(familia, completar_parametros(familia, ajustados))
        inicio = perf_counter()
        pipeline.fit(train[FEATURES], train[TARGET])
        tiempo = perf_counter() - inicio
        metricas = evaluar(validacion[TARGET], pipeline.predict(validacion[FEATURES]))

        with mlflow.start_run(run_name=f"trial-{trial.number:03d}", nested=True):
            mlflow.log_params(ajustados)
            mlflow.log_metrics(
                {f"tuning_{k}": v for k, v in metricas.items()}
                | {"training_time_seconds": tiempo}
            )
            mlflow.set_tags(
                TAGS_PROYECTO
                | {"role": "tuning-trial", "family": familia, "trial_number": str(trial.number)}
            )
        trial.set_user_attr("r2", metricas["r2"])
        return metricas["rmsle"]

    with mlflow.start_run(run_name=f"hpo-{nombre_familia(familia)}") as parent:
        mlflow.log_input(
            mlflow.data.from_pandas(train, targets=TARGET, name="tee-train"),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(validacion, targets=TARGET, name="tee-validacion"),
            context="tuning-validation",
        )
        mlflow.log_params(
            {
                "optimizer": sampler,
                "n_trials": n_trials,
                "sampler_seed": SEED,
                "objective_metric": "tuning_rmsle",
                "direction": "minimize",
            }
        )
        mlflow.set_tags(TAGS_PROYECTO | {"role": "tuning-study", "family": familia})

        estudio.optimize(objective, n_trials=n_trials)

        mejor = estudio.best_trial
        mlflow.log_params({f"best_{k}": v for k, v in mejor.params.items()})
        mlflow.log_param("best_trial_number", mejor.number)
        mlflow.log_metrics(
            {"best_tuning_rmsle": mejor.value, "best_tuning_r2": mejor.user_attrs["r2"]}
        )

    return {
        "params": completar_parametros(familia, mejor.params),
        "best_trial": mejor.number,
        "rmsle_validacion": mejor.value,
        "r2_validacion": mejor.user_attrs["r2"],
        "n_trials": len(estudio.trials),
        "hpo_run_id": parent.info.run_id,
    }
