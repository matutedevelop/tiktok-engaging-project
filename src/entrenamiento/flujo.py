"""Flow de Prefect: datos -> validación -> features -> HPO -> entrenamiento -> test -> Registry.

Ejecución desde la raíz del repositorio:

    uv run python -m entrenamiento.flujo --hacer-tuning --identificador primera-corrida
    uv run python -m entrenamiento.flujo --sin-tuning --identificador sin-hpo
"""

import json
from pathlib import Path
from time import perf_counter

import mlflow
import mlflow.sklearn
import pandas as pd
import typer
from mlflow import MlflowClient
from mlflow.models import infer_signature
from prefect import flow, get_run_logger, task
from prefect.cache_policies import NO_CACHE
from sklearn.model_selection import train_test_split

from entrenamiento.configuracion import (
    ALIASES,
    COLUMNAS_ENTRADA,
    COLUMNAS_LECTURA,
    FAMILIAS,
    FEATURES,
    N_FILAS,
    RUTA_DATOS,
    RUTA_PARAMETROS,
    SEED,
    TAGS_PROYECTO,
    TARGET,
    TARGET_RAW,
    configurar_mlflow,
    nombre_familia,
    nombre_modelo_registrado,
)
from entrenamiento.features import (
    construir_pipeline,
    normalizar_entrada,
    pipeline_de_inferencia,
    preparar_features,
)
from entrenamiento.hpo import ajustar_familia, evaluar

MIN_FILAS = 1_000


# Las tasks reciben DataFrames grandes: calcular una llave de caché con ellos sería caro
# y no queremos reutilizar resultados entre ejecuciones, así que se desactiva la caché.
@task(name="configurar-mlflow", task_run_name="configurar-mlflow", cache_policy=NO_CACHE)
def configurar_destino() -> dict[str, str]:
    destino = configurar_mlflow()
    get_run_logger().info(
        "MLflow -> %s / %s, experimento %s",
        destino["tracking_uri"],
        destino["registry_uri"],
        destino["experiment_name"],
    )
    return destino


@task(name="cargar-datos", task_run_name="cargar-datos", cache_policy=NO_CACHE)
def cargar_datos(ruta: Path, n_filas: int) -> pd.DataFrame:
    logger = get_run_logger()
    if not ruta.exists():
        raise FileNotFoundError(
            f"No existe {ruta.name}; genera data/interim con "
            "`uv run python notebooks/02-preparacion-datos.py sample`"
        )
    df = pd.read_parquet(ruta, columns=COLUMNAS_LECTURA)
    logger.info("%s tiene %s filas", ruta.name, f"{len(df):,}")
    muestra = df.sample(n=min(n_filas, len(df)), random_state=SEED).reset_index(drop=True)
    logger.info("Submuestra de %s filas (random_state=%s)", f"{len(muestra):,}", SEED)
    return muestra


@task(name="validar-datos", task_run_name="validar-datos", cache_policy=NO_CACHE)
def validar_datos(df: pd.DataFrame) -> pd.DataFrame:
    """Un schema inválido no se corrige reintentando: falla sin retries."""
    logger = get_run_logger()
    faltantes = sorted(set(COLUMNAS_LECTURA) - set(df.columns))
    if faltantes:
        raise ValueError(f"Faltan columnas: {faltantes}")
    if len(df) < MIN_FILAS:
        raise ValueError(f"Sólo hay {len(df)} filas; se requieren al menos {MIN_FILAS}")
    target = pd.to_numeric(df[TARGET_RAW], errors="coerce")
    if target.isna().any():
        raise ValueError(f"{int(target.isna().sum())} filas sin {TARGET_RAW}")
    if (target < 0).any():
        raise ValueError(f"{TARGET_RAW} tiene valores negativos")
    nulos = df.isna().sum()
    for columna, n in nulos[nulos > 0].items():
        logger.warning("%s tiene %s nulos", columna, n)
    logger.info("Schema válido: %s columnas, %s filas", len(df.columns), f"{len(df):,}")
    return df


@task(name="preparar-features", task_run_name="preparar-features", cache_policy=NO_CACHE)
def preparar(df: pd.DataFrame) -> pd.DataFrame:
    datos = preparar_features(df)
    get_run_logger().info(
        "%s features + target %s; media de y = %.3f", len(FEATURES), TARGET, datos[TARGET].mean()
    )
    return datos


@task(name="dividir-datos", task_run_name="dividir-datos", cache_policy=NO_CACHE)
def dividir(datos: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """80/10/10 estratificado por deciles de y, igual que en 03-experimentos."""
    estratos = pd.qcut(datos[TARGET], q=10, labels=False, duplicates="drop")
    desarrollo, test = train_test_split(
        datos, test_size=0.10, stratify=estratos, random_state=SEED
    )
    estratos_dev = pd.qcut(desarrollo[TARGET], q=10, labels=False, duplicates="drop")
    train, validacion = train_test_split(
        desarrollo, test_size=1 / 9, stratify=estratos_dev, random_state=SEED
    )
    get_run_logger().info(
        "train=%s, validacion=%s, test=%s", len(train), len(validacion), len(test)
    )
    return {"train": train, "validacion": validacion, "desarrollo": desarrollo, "test": test}


def leer_parametros(ruta: Path) -> dict[str, dict]:
    if not ruta.exists():
        return {}
    return json.loads(ruta.read_text(encoding="utf-8"))


def guardar_parametros(ruta: Path, parametros: dict[str, dict]) -> None:
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(parametros, indent=2), encoding="utf-8")


@task(name="obtener-parametros", task_run_name="parametros-{familia}", cache_policy=NO_CACHE)
def obtener_parametros(
    familia: str,
    hacer_tuning: bool,
    train: pd.DataFrame,
    validacion: pd.DataFrame,
    ruta_parametros: Path,
    n_trials: int | None = None,
) -> dict:
    """Con HPO: corre el estudio y guarda el resultado. Sin HPO: reutiliza el JSON."""
    logger = get_run_logger()
    guardados = leer_parametros(ruta_parametros)
    if hacer_tuning:
        logger.info("HPO activado para %s", familia)
        guardados[familia] = ajustar_familia(familia, train, validacion, n_trials)
        guardar_parametros(ruta_parametros, guardados)
        logger.info(
            "%s: %s trials, mejor RMSLE de validación = %.4f (trial %s)",
            familia,
            guardados[familia]["n_trials"],
            guardados[familia]["rmsle_validacion"],
            guardados[familia]["best_trial"],
        )
        return guardados[familia]
    if familia not in guardados:
        raise FileNotFoundError(
            f"Faltan parámetros de {familia} en {ruta_parametros.name}; usa --hacer-tuning"
        )
    logger.info("Reutilizando parámetros de %s desde %s", familia, ruta_parametros.name)
    return guardados[familia]


@task(name="entrenar-modelo", task_run_name="entrenar-{familia}", cache_policy=NO_CACHE)
def entrenar_modelo_final(familia: str, parametros: dict, desarrollo: pd.DataFrame) -> dict:
    """Reentrena con train + validación; el preprocesamiento se ajusta dentro del Pipeline."""
    pipeline = construir_pipeline(familia, parametros)
    inicio = perf_counter()
    pipeline.fit(desarrollo[FEATURES], desarrollo[TARGET])
    tiempo = perf_counter() - inicio
    get_run_logger().info("%s entrenado en %.1f s", familia, tiempo)
    return {"pipeline": pipeline, "training_time_seconds": tiempo}


@task(name="evaluar-modelo", task_run_name="evaluar-{familia}", cache_policy=NO_CACHE)
def evaluar_modelo(familia: str, entrenamiento: dict, test: pd.DataFrame) -> dict[str, float]:
    """Única vez que se consulta el test, después de fijar parámetros y entrenar."""
    metricas = evaluar(test[TARGET], entrenamiento["pipeline"].predict(test[FEATURES]))
    get_run_logger().info(
        "%s en test: RMSLE=%.4f, R2=%.4f, Spearman=%.4f",
        familia,
        metricas["rmsle"],
        metricas["r2"],
        metricas["spearman"],
    )
    return metricas


@task(name="registrar-modelo", task_run_name="registrar-{familia}", cache_policy=NO_CACHE)
def registrar_modelo_final(
    familia: str,
    hpo: dict,
    entrenamiento: dict,
    metricas: dict[str, float],
    desarrollo: pd.DataFrame,
    test: pd.DataFrame,
    ejemplo: pd.DataFrame,
    modelo_registrado: str,
    identificador: str,
) -> dict:
    """Run `<familia>-final` dentro de `hpo-<familia>` y nueva versión en Unity Catalog."""
    modelo = pipeline_de_inferencia(entrenamiento["pipeline"])
    firma = infer_signature(ejemplo, modelo.predict(ejemplo))

    with mlflow.start_run(
        run_name=f"{nombre_familia(familia)}-final", parent_run_id=hpo["hpo_run_id"]
    ) as run:
        mlflow.log_input(
            mlflow.data.from_pandas(desarrollo, targets=TARGET, name="tee-desarrollo"),
            context="training",
        )
        mlflow.log_input(
            mlflow.data.from_pandas(test, targets=TARGET, name="tee-test"),
            context="final-evaluation",
        )
        mlflow.log_params(hpo["params"])
        mlflow.log_metrics(
            {f"test_{k}": v for k, v in metricas.items()}
            | {"training_time_seconds": entrenamiento["training_time_seconds"]}
        )
        mlflow.set_tags(
            TAGS_PROYECTO
            | {
                "role": "model-final",
                "family": familia,
                "best_trial_number": str(hpo["best_trial"]),
                "prefect_run": identificador,
            }
        )
        info = mlflow.sklearn.log_model(
            modelo,
            name="model",
            input_example=ejemplo,
            signature=firma,
            serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE,
        )

    version = mlflow.register_model(f"models:/{info.model_id}", name=modelo_registrado)
    cliente = MlflowClient()
    etiquetas = {
        "family": familia,
        "final_run_id": run.info.run_id,
        "hpo_run_id": hpo["hpo_run_id"],
        "test_rmsle": f"{metricas['rmsle']:.6f}",
        "test_r2": f"{metricas['r2']:.6f}",
        "prefect_run": identificador,
    }
    for clave, valor in etiquetas.items():
        cliente.set_model_version_tag(modelo_registrado, version.version, clave, valor)
    cliente.update_model_version(
        modelo_registrado,
        version.version,
        description=(
            f"{familia} entrenado con train+validación ({len(desarrollo):,} filas) y evaluado "
            f"en test: RMSLE={metricas['rmsle']:.4f}, R2={metricas['r2']:.4f}. "
            f"Run {nombre_familia(familia)}-final ({run.info.run_id})."
        ),
    )
    get_run_logger().info("%s registrado como versión %s", familia, version.version)
    return {
        "familia": familia,
        "version": str(version.version),
        "test_rmsle": metricas["rmsle"],
        "test_r2": metricas["r2"],
        "test_spearman": metricas["spearman"],
        "rmsle_validacion": hpo["rmsle_validacion"],
        "training_time_seconds": entrenamiento["training_time_seconds"],
        "final_run_id": run.info.run_id,
        "model_id": info.model_id,
    }


@task(name="asignar-aliases", task_run_name="asignar-aliases", cache_policy=NO_CACHE)
def asignar_aliases(resultados: list[dict], modelo_registrado: str) -> pd.DataFrame:
    """Menor RMSLE de test -> champion; no se fija de antemano qué familia gana."""
    ranking = pd.DataFrame(resultados).sort_values("test_rmsle").reset_index(drop=True)
    ranking.insert(0, "posicion", range(1, len(ranking) + 1))
    ranking["alias"] = list(ALIASES[: len(ranking)])
    cliente = MlflowClient()
    for fila in ranking.itertuples(index=False):
        cliente.set_registered_model_alias(modelo_registrado, fila.alias, fila.version)
        cliente.set_model_version_tag(modelo_registrado, fila.version, "alias", fila.alias)
    get_run_logger().info(
        "champion = %s (versión %s)", ranking.loc[0, "familia"], ranking.loc[0, "version"]
    )
    return ranking


@task(name="verificar-champion", task_run_name="verificar-champion", cache_policy=NO_CACHE)
def verificar_champion(modelo_registrado: str, ejemplo: pd.DataFrame) -> dict:
    """Carga `@champion` igual que lo hará la API y comprueba que predice."""
    version = MlflowClient().get_model_version_by_alias(modelo_registrado, "champion")
    modelo = mlflow.pyfunc.load_model(f"models:/{modelo_registrado}@champion")
    predicciones = modelo.predict(ejemplo)
    get_run_logger().info(
        "@champion -> versión %s (%s); predice %s filas",
        version.version,
        version.tags.get("family"),
        len(predicciones),
    )
    return {
        "version": str(version.version),
        "familia": version.tags.get("family"),
        "log_play_count": [float(p) for p in predicciones],
    }


@flow(name="tee-entrenamiento", flow_run_name="tee-{identificador}", log_prints=True)
def flujo_entrenamiento(
    identificador: str = "local",
    hacer_tuning: bool = False,
    n_filas: int = N_FILAS,
    ruta_datos: Path = RUTA_DATOS,
    ruta_parametros: Path = RUTA_PARAMETROS,
    n_trials: int | None = None,
    familias: tuple[str, ...] = FAMILIAS,
) -> dict:
    configurar_destino()
    modelo_registrado = nombre_modelo_registrado()

    crudos = validar_datos(cargar_datos(ruta_datos, n_filas))
    particiones = dividir(preparar(crudos))
    # Ejemplo de entrada con columnas crudas, tomado del test: es lo que recibirá la API
    ejemplo = normalizar_entrada(crudos.loc[particiones["test"].index[:5], COLUMNAS_ENTRADA])

    resultados = []
    for familia in familias:
        hpo = obtener_parametros(
            familia,
            hacer_tuning,
            particiones["train"],
            particiones["validacion"],
            ruta_parametros,
            n_trials,
        )
        entrenamiento = entrenar_modelo_final(familia, hpo["params"], particiones["desarrollo"])
        metricas = evaluar_modelo(familia, entrenamiento, particiones["test"])
        resultados.append(
            registrar_modelo_final(
                familia,
                hpo,
                entrenamiento,
                metricas,
                particiones["desarrollo"],
                particiones["test"],
                ejemplo,
                modelo_registrado,
                identificador,
            )
        )

    ranking = asignar_aliases(resultados, modelo_registrado)
    verificacion = verificar_champion(modelo_registrado, ejemplo)
    print(ranking.drop(columns=["final_run_id", "model_id"]).to_string(index=False))
    return {
        "modelo_registrado": modelo_registrado,
        "ranking": ranking,
        "verificacion": verificacion,
    }


app = typer.Typer(add_completion=False)


@app.command()
def main(
    identificador: str = typer.Option("local", help="Nombre anónimo de la corrida"),
    hacer_tuning: bool = typer.Option(
        False, "--hacer-tuning/--sin-tuning", help="Correr HPO o reutilizar el JSON"
    ),
    n_filas: int = typer.Option(N_FILAS, help="Tamaño de la submuestra"),
    n_trials: int | None = typer.Option(None, help="Sobrescribe los trials de cada familia"),
) -> None:
    flujo_entrenamiento(
        identificador=identificador,
        hacer_tuning=hacer_tuning,
        n_filas=n_filas,
        n_trials=n_trials,
    )


if __name__ == "__main__":
    app()
