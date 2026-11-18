import marimo

__generated_with = "0.24.2"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 09 — Despliegue en AWS EC2

    La API y la interfaz se desplegaron en una instancia EC2 con las mismas imágenes y el mismo
    `docker-compose.yaml` de [`08-contenedores`](08-contenedores.ipynb), y se obtuvo una
    predicción desde otra computadora.

    En este proyecto "AWS" significa **cómputo**: una máquina virtual que corre los contenedores.
    El código no usa boto3, S3 ni IAM. El modelo `champion` se sigue cargando desde
    Databricks (MLflow + Unity Catalog) con un `.env` que solo existe en el servidor.

    ```
    otra computadora ──http :8501──▶ EC2 [ui 0.0.0.0:8501] ──127.0.0.1:8000──▶ [api 127.0.0.1:8000] ──HTTPS──▶ Databricks
                       (grupo de seguridad: solo IPs autorizadas)                (no expuesta)
    ```

    El despliegue lo hizo Owen Loza con su usuario de la consola AWS del curso, su llave `.pem`
    y su grupo de seguridad. En las capturas se ocultaron IPs, identificadores de instancia y de
    grupo de seguridad, la cuenta y el hostname.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Instancia

    | Concepto | Valor |
    |---|---|
    | Región / zona | `us-east-1` (N. Virginia) / `us-east-1a` |
    | Tipo | `t3.medium` (2 vCPU, 3.7 GiB de RAM, sin swap) |
    | Sistema operativo | Ubuntu 24.04.5 LTS, usuario `ubuntu` |
    | Disco raíz (EBS) | 19 GB, 13 GB libres antes del build |
    | Docker / Compose | Docker 29.9.3 / Docker Compose v5.7.0 |
    | Carpeta del proyecto | `~/tiktok-engaging-project`, separada de otros trabajos de la instancia |
    | IP pública | cambia en cada *Start*; se escribe como `IP_PUBLICA` |

    Es la misma instancia de la Clase 20: se encendió con *Start* y no se creó otra. `t3.medium`
    alcanzó: el build y la carga del modelo terminaron sin quedarse sin memoria ni disco.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Reglas de acceso (grupo de seguridad)

    | Puerto | Protocolo | Origen | Para qué |
    |---|---|---|---|
    | 22 | TCP (SSH) | *My IP* (`/32`) | administrar la instancia |
    | 8501 | TCP | *My IP* (`/32`) y la IP `/32` de cada persona que prueba | la interfaz |
    | 8000 | — | **sin regla** | la API no se expone |

    - Ninguna regla usa `0.0.0.0/0`. La regla del 8501 venía de la Clase 20 abierta a `0.0.0.0/0`
      y se restringió a *My IP* antes de desplegar.
    - Hay **dos** barreras para el 8000: el grupo de seguridad no lo abre y la API escucha en
      `127.0.0.1`. Aunque se abriera el puerto, nadie de fuera podría conectarse.
    - Si la IP de quien administra cambia (otra red), hay que volver a elegir *My IP* en las dos
      reglas; si no, SSH y la interfaz dan *timeout*.

    | Evidencia | Archivo |
    |---|---|
    | Instancia `Running` con `3/3 checks passed` | [`evidencia/instancia-running.png`](../evidencia/instancia-running.png) |
    | Reglas de entrada (22 y 8501 desde una sola IP, sin 8000) | [`evidencia/instancia-reglas-entrada.png`](../evidencia/instancia-reglas-entrada.png) |

    ![Instancia](../evidencia/instancia-running.png)
    ![Reglas](../evidencia/instancia-reglas-entrada.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Pasos reproducibles

    **Consola AWS.** Región `us-east-1` → EC2 → *Instances* → la instancia → *Instance state →
    Start*. Esperar `Running` y los *status checks*, anotar la IPv4 pública y revisar las reglas
    de la sección 2.

    **Conexión** (Git Bash en la computadora local):

    ```bash
    ssh -i ~/.ssh/<llave>.pem ubuntu@IP_PUBLICA
    whoami; cat /etc/os-release | head -3
    df -h /; free -h
    ```

    **Docker** (solo si `sudo docker --version && sudo docker compose version` falla): repositorio
    oficial de Docker para Ubuntu `noble` y
    `sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin`.
    La comprobación es `sudo docker run --rm hello-world`. En esta instancia ya estaba instalado.

    **Código.** El repositorio es público: se clona sin token. En la EC2 solo se lee el repo
    (clone y pull); commits y evidencias se suben desde la computadora local.

    ```bash
    cd ~
    git clone https://github.com/matutedevelop/tiktok-engaging-project.git
    cd tiktok-engaging-project
    git switch feat/tercera-entrega     # después del merge: main
    git log --oneline -1
    ```

    **Configuración privada.** El `.env` se escribe a mano en el servidor; nunca pasa por Git ni
    por la imagen.

    ```bash
    cp .env.example .env
    chmod 600 .env
    nano .env                                   # solo DATABRICKS_HOST y DATABRICKS_TOKEN
    cp src/frontend/.env.example src/frontend/.env
    git check-ignore .env src/frontend/.env     # imprime las dos rutas
    git status --short                          # vacío
    grep -c '<' .env                            # 0: no quedó ningún <placeholder>
    sed 's/=.*/=***/' .env | grep -v '^#'       # nombres de variables sin valores
    ```

    El `.env` queda con permisos `-rw-------`: solo el usuario `ubuntu` lo puede leer. Se
    recomienda un token de Databricks propio para el servidor, con vencimiento corto.

    **Construir y arrancar** (la terminal queda mostrando los logs):

    ```bash
    sudo docker compose config --quiet
    sudo docker compose up --build
    ```
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Despliegue

    - **Build.** Las dos imágenes se construyeron en paralelo en ~155 s: `pip install` tardó 94 s
      en la API y 31 s en la interfaz. Terminaron en `Built` y los contenedores en `Created`.
    - **Carga del modelo.** La API resolvió el alias y cargó
      `workspace.tiktok-engaging-estimator.pcd_otono_2026_tiktok_engaging_estimator@champion -> version 3 (lightgbm)`,
      y quedó en `Uvicorn running on http://127.0.0.1:8000`.

    | Evidencia | Archivo |
    |---|---|
    | Build en EC2 | [`evidencia/despliegue-ec2-build.png`](../evidencia/despliegue-ec2-build.png) |
    | Modelo cargado y API en `127.0.0.1:8000` | [`evidencia/despliegue-ec2-modelo-cargado.png`](../evidencia/despliegue-ec2-modelo-cargado.png) |

    ![Build en EC2](../evidencia/despliegue-ec2-build.png)
    ![Modelo cargado](../evidencia/despliegue-ec2-modelo-cargado.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Comprobaciones

    ### Dentro del servidor (segunda terminal SSH)

    ```bash
    curl -i http://127.0.0.1:8000/health
    curl -i -X POST http://127.0.0.1:8000/api/v1/predicciones -H "Content-Type: application/json" -d '{}'
    sudo docker compose ps
    ```

    | Comprobación | Resultado |
    |---|---|
    | `GET /health` | `200 OK`, `"alias":"champion","version":"3","familia":"lightgbm"` |
    | `POST` con `{}` | `422 Unprocessable Entity`, con los 19 campos requeridos marcados `missing` y sin predicción |
    | `docker compose ps` | `tee-api-container` y `tee-ui-container` en `Up` |

    La entrada inválida se probó **desde dentro** del servidor, sin exponer la API.

    ### Desde fuera del servidor

    | Destino | Resultado |
    |---|---|
    | `http://IP_PUBLICA:8000/health` | sin respuesta (*timeout*): la API no es alcanzable desde Internet |
    | `http://IP_PUBLICA:8501/_stcore/health` | `200` |

    ### Predicción remota

    Desde el navegador de otra computadora, en `http://IP_PUBLICA:8501`:

    | | Valor |
    |---|---|
    | Estado en la barra lateral | *Disponible · modelo versión 3 (lightgbm)* |
    | Entrada | el ejemplo por defecto: `desc = "Amazon!!!! #amazonfinds #affordablefashion"`, `challenges = [amazonfinds, affordablefashion]`, `duration = 28`, `vq_score = 74.62`, `music_duration = 44.0`, `poi_category = "Place and Address"`, resto de campos del ejemplo válido de la API |
    | Salida | **10,905 reproducciones**, orden de magnitud 10⁴, rango típico 1,393 – 85,305 |
    | Versión servida | `workspace.tiktok-engaging-estimator.pcd_otono_2026_tiktok_engaging_estimator`, alias `champion`, versión 3, familia `lightgbm` |

    Es la misma predicción que en local con y sin contenedores: la versión servida en EC2 es la
    misma que el `champion` del Registry.

    ### Logs

    `sudo docker compose logs --tail 30 api` muestra la secuencia completa del despliegue:

    ```
    22:56:02  tee.api INFO Modelo cargado: …@champion -> version 3 (lightgbm)
              INFO: Application startup complete.
              INFO: 127.0.0.1 - "GET /health HTTP/1.1" 200 OK
              INFO: 127.0.0.1 - "POST /api/v1/predicciones HTTP/1.1" 422 Unprocessable Entity   ← curl con {}
    23:01:39  tee.api INFO Prediccion version 3: 10905 reproducciones                          ← navegador remoto
              INFO: 127.0.0.1 - "POST /api/v1/predicciones HTTP/1.1" 200 OK
    ```

    (horas en UTC). Todas las peticiones a la API vienen de `127.0.0.1`: la predicción remota
    entró por la interfaz y la interfaz llamó a la API dentro de la misma máquina. La línea
    `Prediccion version 3: 10905` corresponde a la salida que se vio en el navegador.

    | Evidencia | Archivo |
    |---|---|
    | Predicción remota con versión | [`evidencia/despliegue-ec2-prediccion-remota.png`](../evidencia/despliegue-ec2-prediccion-remota.png) |
    | `/health` 200 y entrada inválida 422 dentro de EC2 | [`evidencia/despliegue-ec2-entrada-invalida.png`](../evidencia/despliegue-ec2-entrada-invalida.png) |
    | Logs de la API con la predicción | [`evidencia/despliegue-ec2-logs.png`](../evidencia/despliegue-ec2-logs.png) |

    ![Predicción remota](../evidencia/despliegue-ec2-prediccion-remota.png)
    ![Entrada inválida](../evidencia/despliegue-ec2-entrada-invalida.png)
    ![Logs](../evidencia/despliegue-ec2-logs.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Problemas encontrados

    | Problema | Causa | Solución |
    |---|---|---|
    | La instancia no aparecía en la lista | filtro `Instance state = running` y la instancia estaba en `Stopped` | quitar el filtro y hacer *Start* |
    | 8501 abierto a `0.0.0.0/0` | regla heredada de la Clase 20 | cambiar el origen a *My IP* |
    | `git push` con 403 desde la computadora local | la cuenta de GitHub activa no era colaboradora del repo | cambiar de cuenta en `gh auth switch` antes del push |

    Otros síntomas posibles: `Connection timed out` en SSH (IP, estado o regla 22),
    `Permission denied (publickey)` (usuario o llave), la página no abre (regla 8501 o IP de
    origen), la página abre pero no predice (la API aún no carga el modelo; revisar
    `docker compose logs api` y que `API_URL` use `127.0.0.1`).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. Cierre y costos

    ```bash
    # Ctrl+C en la terminal del up
    sudo docker compose down
    sudo docker compose ps        # vacío
    exit
    ```

    Después, en la consola: *Instance state → **Stop*** (no *Terminate*) hasta ver `Stopped`.

    - Los contenedores salieron con `exit code 0` y la API cerró ordenadamente
      (`Application shutdown complete`); `down` los eliminó.
    - **Stop** apaga la máquina: deja de cobrarse el cómputo (`t3.medium`, del orden de
      USD 0.04 por hora en `us-east-1`). El despliegue completo estuvo encendido menos de una hora.
    - **El volumen EBS se conserva y se sigue cobrando** aunque la instancia esté detenida (del
      orden de USD 0.08 por GB al mes en `gp3`; ≈ USD 1.6 al mes para 20 GB). El propio diálogo de
      *Stop* lo advierte. A cambio, el repositorio clonado, el `.env` y las capas de Docker siguen
      ahí para la verificación después del merge.
    - **Terminate** borraría la instancia y su disco; no se usa.
    - El token de Databricks del servidor se revoca al terminar la verificación final.

    | Evidencia | Archivo |
    |---|---|
    | `down` y `ps` vacío | [`evidencia/cierre-docker-down.png`](../evidencia/cierre-docker-down.png) |
    | Diálogo de *Stop* con la advertencia de costos de EBS | [`evidencia/cierre-stop-costos.png`](../evidencia/cierre-stop-costos.png) |
    | Instancia en `Stopped` | [`evidencia/cierre-instancia-stopped.png`](../evidencia/cierre-instancia-stopped.png) |

    ![Docker down](../evidencia/cierre-docker-down.png)
    ![Stop](../evidencia/cierre-stop-costos.png)
    ![Stopped](../evidencia/cierre-instancia-stopped.png)
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. Limitaciones

    - **Sin HTTPS.** La interfaz se sirve por HTTP plano en el puerto 8501 (el navegador la marca
      como "No es seguro"). Para uso real haría falta un dominio, un proxy inverso y un
      certificado.
    - **IP pública variable.** Cambia en cada *Start*; una Elastic IP la fijaría, con costo
      mientras la instancia está detenida.
    - **Acceso por lista de IPs.** Cada persona que prueba necesita su regla `/32`; no hay
      autenticación en la aplicación.
    - **Arranque manual.** El servicio corre en primer plano con `docker compose up`; si se cierra
      la sesión SSH o se reinicia la instancia, hay que levantarlo otra vez. `up -d` con
      `restart: unless-stopped` lo resolvería.
    - **Dependencia de Databricks.** Si el token vence o Databricks no responde, la API arranca en
      estado degradado (`/health` 503) y no predice.
    - **Una sola instancia**, sin balanceo ni monitoreo más allá de los logs de Docker.
    """)
    return


if __name__ == "__main__":
    app.run()
