# IMPACTO DEL SESGO EN LOS DATOS SOBRE EL RENDIMIENTO DE ALGORITMOS DE INTELIGENCIA ARTIFICIAL: ANÁLISIS, MITIGACIÓN Y CONSECUENCIAS ÉTICA

Descarga del dataset, ejecución del pipeline y generación de resultados

## Origen de los datos

El conjunto de datos utilizado procede de The Public Jira Dataset, un dataset público construido a partir de repositorios Jira accesibles en Internet. La publicación original describe una colección de 16 repositorios públicos de Jira, 1.822 proyectos, 2,7 millones de issues, 32 millones de cambios, 9 millones de comentarios y 1 millón de enlaces entre issues. El dataset se distribuye como un volcado de MongoDB, por lo que no se descarga inicialmente como una tabla CSV lista para entrenamiento, sino como una base documental que debe restaurarse y transformar posteriormente ([Zenodo](https://zenodo.org/records/15719919)).

En este TFM se trabaja con una versión anonimizada del dataset. Esta aclaración es importante porque el dataset público permite estudiar patrones operativos, calidad de datos, tiempos de resolución y comportamiento por subgrupos, pero no permite analizar atributos sensibles clásicos como género, edad, nacionalidad o características personales protegidas. En un entorno Jira real sí podrían existir campos de usuario, equipo, responsable, reporter, cliente o unidad organizativa; por tanto, cualquier aplicación real del marco debería incorporar controles adicionales de privacidad, minimización de datos y revisión ética.

## Descarga de The Public Jira Dataset

La descarga debe realizarse desde la página oficial del dataset en Zenodo. En la versión actual, Zenodo proporciona el archivo comprimido `ThePublicJiraDataset.zip`, que contiene el volcado de MongoDB y materiales asociados al dataset ([Zenodo](https://zenodo.org/records/15719919)).

Pasos recomendados:

1. Acceder a la página oficial del dataset en Zenodo.
2. Descargar el archivo comprimido del dataset.
3. Guardar el archivo en una carpeta local destinada a datos originales:

```text
datasets/
└── ThePublicJiraDataset/
    └── ThePublicJiraDataset.zip
```

4. Descomprimir el archivo:

```bash
cd datasets/ThePublicJiraDataset
unzip ThePublicJiraDataset.zip
```

Después de la descompresión, debe localizarse la carpeta que contiene el volcado de MongoDB. 

## Requisitos previos

Antes de restaurar y procesar el dataset, el equipo debe tener instalados los siguientes componentes:

| Componente | Finalidad |
|---|---|
| Python 3 | Ejecución de scripts de extracción, preparación, modelado y análisis |
| Entorno virtual de Python | Aislamiento de dependencias del proyecto |
| MongoDB Server | Restauración y consulta del volcado original del dataset |
| MongoDB Database Tools | Uso de comandos como `mongorestore` |
| Librerías Python | Tratamiento de datos, modelado, métricas, gráficos y conexión con MongoDB |
| Power BI Desktop | Visualización posterior de los CSV generados por el pipeline |


## Preparación del entorno de Python

Desde la carpeta principal del proyecto donde se encuentran las carpetas `src`, `config`, `data` y `reports`, se recomienda crear un entorno virtual:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
```

Después se instalan las dependencias. Se puede utilizar `requirements.txt`:

```bash
pip install -r requirements.txt
```

Estas librerías permiten cubrir las tareas principales del pipeline: conexión con MongoDB, transformación tabular, análisis de calidad, entrenamiento de modelos, cálculo de métricas, experimentos de degradación, auditoría de sesgos, generación de salidas para Power BI y trazabilidad.

## Estructura de carpetas utilizada

La estructura de carpetas recomendada es la siguiente:

```text
tfm-jira-pipeline/
├── config/
│   └── config.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── powerbi/
├── reports/
├── src/
└── .venv/
```

La carpeta `data/raw` almacena el CSV inicial extraído desde MongoDB. La carpeta `data/processed` contiene los datasets preparados para modelado. La carpeta `reports` recoge las métricas y salidas analíticas generadas por los scripts. La carpeta `data/powerbi` organiza los archivos destinados a visualización en Power BI.


Esta organización es importante porque permite relacionar cada resultado con una fase concreta del pipeline y evita que las salidas queden dispersas.

## Restauración del volcado de MongoDB

The Public Jira Dataset se distribuye como un volcado de MongoDB, por lo que el primer paso técnico no es entrenar modelos, sino restaurar la base de datos. La ruta exacta dependerá de la ubicación donde se haya descomprimido el dataset.

Ejemplo de restauración:

```bash
mongorestore --drop --db JiraReposAnon "[RUTA_A_LA_CARPETA_DEL_DUMP]"
```

La restauración puede tardar debido al tamaño del dataset. Una vez finalizada, debe comprobarse que la base de datos está disponible.

## Inspección inicial de MongoDB

Después de restaurar el volcado, se accede a la consola de MongoDB:

```bash
mongosh
```

Dentro de la consola se ejecutan comandos de comprobación:

```javascript
show dbs
use JiraReposAnon
show collections
```

Para contar documentos en una colección:

```javascript
db.getCollection("[NOMBRE_COLECCION]").countDocuments()
```

Para visualizar un documento de ejemplo:

```javascript
db.getCollection("[NOMBRE_COLECCION]").findOne()
```

Estas comprobaciones permiten verificar tres aspectos: que la base de datos se ha restaurado, que existen colecciones disponibles y que los documentos tienen la estructura esperada. Si esta comprobación falla, no debe ejecutarse todavía el pipeline, porque el script de extracción depende de que MongoDB esté correctamente restaurado.


## Configuración del pipeline

El archivo `config/config.yaml` centraliza los parámetros de ejecución. Esta configuración permite evitar rutas dispersas en el código y facilita repetir el pipeline en otro equipo.

Ejemplo de configuración mínima:

```yaml
data:
  raw_dataset: data/raw/jira_issues.csv
  modeling_dataset: data/processed/jira_issues_prepared.csv
  cleaned_dataset: data/processed/jira_issues_cleaned.csv

mongodb:
  database: JiraReposAnon
  collection: "[NOMBRE_COLECCION]"

modeling:
  target: late_resolution
  random_state: 42
  test_size: 0.2
  threshold: 0.25

outputs:
  reports_dir: reports
  powerbi_dir: data/powerbi
```

Durante la ejecución del TFM se detectó que algunos scripts necesitaban localizar explícitamente el dataset tabular preparado. Por ello, la clave `data.modeling_dataset` debe estar definida. Si falta, scripts como `bias_fairness_audit.py` pueden generar errores al no encontrar el CSV preparado.

## Orden general del pipeline

El pipeline debe ejecutarse siguiendo un orden lógico. Primero se extraen y preparan los datos. Después se realiza el diagnóstico de calidad. A continuación se entrenan modelos y se calculan métricas. Posteriormente se ejecutan experimentos de degradación, limpieza, drift, reentrenamiento, estabilidad, sesgos y mitigación. Finalmente se generan salidas para Power BI y artefactos de auditoría.



**Orden recomendado de ejecución de scripts y salidas principales**

| Orden | Script | Finalidad | Entrada principal | Salida principal |
|---:|---|---|---|---|
| 1 | `extract_from_mongodb.py` | Extraer issues desde MongoDB y generar CSV inicial | Base MongoDB restaurada | `data/raw/jira_issues.csv` |
| 2 | `prepare_dataset.py` | Transformar datos en tabla analítica | `data/raw/jira_issues.csv` | `data/processed/jira_issues_prepared.csv` |
| 3 | `eda_quality.py` | Analizar calidad inicial, nulos, cardinalidad y distribuciones | Dataset preparado | CSV de calidad en `reports/` |
| 4 | `train_baseline.py` | Entrenar modelos baseline | Dataset preparado | `reports/model_metrics.csv` |
| 5 | `threshold_tuning.py` | Evaluar umbrales de clasificación | Predicciones/modelos baseline | `reports/threshold_metrics.csv` |
| 6 | `advanced_metrics.py` | Calcular métricas avanzadas y calibración | Dataset preparado/modelos | `reports/advanced_model_metrics.csv` |
| 7 | `subgroup_gap_analysis.py` | Calcular métricas por subgrupo y brechas | Métricas/predicciones | CSV de gaps por subgrupo |
| 8 | `degradation_experiments.py` | Simular degradaciones controladas de calidad | Dataset preparado | `reports/degradation_metrics.csv` |
| 9 | `data_cleaning_experiment.py` | Evaluar efecto de limpieza de datos | Dataset preparado | `data/processed/jira_issues_cleaned.csv` |
| 10 | `temporal_drift_experiment.py` | Comparar split aleatorio y temporal | Dataset preparado | CSV de drift temporal |
| 11 | `retraining_experiment.py` | Evaluar modelo estático, acumulado y ventana deslizante | Dataset preparado | `reports/retraining_metrics.csv` |
| 12 | `stability_experiment.py` | Repetir entrenamiento con distintas semillas | Dataset preparado | CSV de estabilidad |
| 13 | `possible_worlds_sensitivity.py` | Simular posibles mundos y sensibilidad | Dataset preparado | CSV de sensibilidad |
| 14 | `bias_fairness_audit.py` | Evaluar algoritmos, subgrupos y fairness operacional | Dataset preparado | CSV de fairness y disparidades |
| 15 | `bias_scenarios_experiment.py` | Simular escenarios controlados de sesgo | Dataset preparado | CSV de escenarios de sesgo |
| 16 | `bias_mitigation_experiment.py` | Evaluar técnicas de mitigación y trade-offs | Dataset preparado | CSV de mitigación |
| 17 | `project_risk_proxy_export.py` | Generar proxies de riesgo de gestión de proyectos | Dataset preparado | CSV de proxies de riesgo |
| 18 | `monitoring_kpis_export.py` | Preparar KPIs para Power BI | CSV generados en `reports/` | `data/powerbi/*.csv` |
| 19 | `audit_manifest.py` | Generar inventario, hashes y trazabilidad | Scripts y resultados | Archivos de auditoría |

Como muestra la tabla, los scripts no son independientes entre sí. Las primeras fases generan datasets que alimentan las siguientes; las fases intermedias generan métricas; y las fases finales reorganizan resultados para visualización y auditoría. Por ello, si se ejecuta un script avanzado sin haber generado antes el dataset preparado o las métricas base, pueden aparecer errores de rutas o archivos inexistentes.

## Comandos de ejecución recomendados

La siguiente secuencia resume los comandos principales. Deben ejecutarse desde la carpeta raíz del proyecto y con el entorno virtual activado.

```bash
python3 src/extract_from_mongodb.py --config config/config.yaml
python3 src/prepare_dataset.py --config config/config.yaml
python3 src/eda_quality.py --config config/config.yaml
python3 src/train_baseline.py --config config/config.yaml
python3 src/threshold_tuning.py --config config/config.yaml
python3 src/advanced_metrics.py --config config/config.yaml
python3 src/subgroup_gap_analysis.py --config config/config.yaml
python3 src/degradation_experiments.py --config config/config.yaml
python3 src/data_cleaning_experiment.py --config config/config.yaml
python3 src/temporal_drift_experiment.py --config config/config.yaml
python3 src/retraining_experiment.py --config config/config.yaml
python3 src/stability_experiment.py --config config/config.yaml --seeds 10
python3 src/possible_worlds_sensitivity.py --config config/config.yaml --worlds 20
python3 src/bias_fairness_audit.py --config config/config.yaml
python3 src/bias_scenarios_experiment.py --config config/config.yaml
python3 src/bias_mitigation_experiment.py --config config/config.yaml
python3 src/project_risk_proxy_export.py --config config/config.yaml
python3 src/monitoring_kpis_export.py --config config/config.yaml
python3 src/audit_manifest.py --config config/config.yaml
```

El orden puede adaptarse si se quiere repetir solo una fase concreta, pero siempre deben existir sus entradas. Por ejemplo, para repetir el análisis de fairness debe existir `data/processed/jira_issues_prepared.csv`; para generar Power BI deben existir previamente los CSV de métricas; y para auditar el pipeline deben existir scripts y resultados ya generados.

## Descripción de cada fase y resultados generados

### Extracción desde MongoDB

Comando:

```bash
python3 src/extract_from_mongodb.py --config config/config.yaml
```

Este script conecta con la base de datos MongoDB restaurada y extrae los documentos de la colección seleccionada. Su función es convertir el origen documental en una primera tabla CSV manejable por el resto del pipeline.

Resultado esperado:

```text
data/raw/jira_issues.csv
```

Este archivo actúa como punto de partida tabular. No es todavía el dataset de modelado, sino una extracción inicial desde la base documental.

### Preparación del dataset analítico

Comando:

```bash
python3 src/prepare_dataset.py --config config/config.yaml
```

Este script transforma el CSV inicial en un dataset preparado para machine learning. Entre otras operaciones, calcula tiempos de resolución, transforma fechas, genera variables derivadas, prepara campos textuales simples y define la variable objetivo `late_resolution`.

Resultado esperado:

```text
data/processed/jira_issues_prepared.csv
```

Este archivo es la entrada principal para los experimentos de modelado, métricas, degradación, subgrupos, sesgos, drift y mitigación.

### Análisis exploratorio y calidad inicial

Comando:

```bash
python3 src/eda_quality.py --config config/config.yaml
```

Este script calcula información descriptiva del dataset: valores nulos, tipos de variables, cardinalidad, distribución de categorías, distribución temporal y proporción de la variable objetivo.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `data_quality_summary.csv` | `reports/` o `data/powerbi/` | Tablas de calidad inicial |
| `category_distributions.csv` | `reports/` o `data/powerbi/` | Distribución de variables categóricas |
| `time_distribution.csv` | `reports/` o `data/powerbi/` | Evolución temporal de registros |

Estas salidas justifican que el dataset presenta problemas reales de completitud, disponibilidad de etiqueta y distribución de clases.

### Entrenamiento baseline

Comando:

```bash
python3 src/train_baseline.py --config config/config.yaml
```

Este script entrena los modelos iniciales utilizados como referencia. En el TFM se emplean principalmente regresión logística y Random Forest. El objetivo es disponer de una primera comparación de rendimiento y calibración.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `model_metrics.csv` | `reports/` | Comparación global de modelos |
| `subgroup_metrics.csv` | `reports/` | Métricas iniciales por subgrupo |
| `calibration_summary.csv` | `reports/` | Resumen inicial de calibración |

Estos resultados sirven como línea base frente a la que se comparan ajustes de umbral, degradaciones, limpieza, drift y mitigaciones.

### Ajuste de umbrales

Comando:

```bash
python3 src/threshold_tuning.py --config config/config.yaml
```

El script evalúa distintos umbrales de clasificación para estudiar el equilibrio entre precision, recall y F1-score. Esta fase es necesaria porque el umbral por defecto de 0,5 no siempre es adecuado en problemas con clase positiva minoritaria.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `threshold_metrics.csv` | `reports/` | Métricas por umbral |
| `threshold_best_f1.csv` | `reports/` | Umbral con mejor F1-score |
| `threshold_recall_60_minimum.csv` | `reports/` | Umbral con recall mínimo |

El resultado permite justificar el uso de un umbral ajustado para Random Forest.

### Métricas avanzadas y calibración

Comando:

```bash
python3 src/advanced_metrics.py --config config/config.yaml
```

Este script amplía la evaluación con métricas derivadas de la matriz de confusión y métricas probabilísticas. Incluye accuracy, error rate, specificity, FPR, FNR, balanced accuracy, Brier score, ECE y calibración por bins.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `advanced_model_metrics.csv` | `reports/` | Métricas avanzadas por modelo |
| `calibration_bins.csv` | `reports/` o `data/powerbi/` | Análisis de calibración por intervalos |
| `prediction_audit.csv` | `reports/` | Auditoría de predicciones individuales |

Estas salidas permiten analizar si el modelo no solo clasifica correctamente, sino también si sus probabilidades son interpretables.

### Análisis por subgrupos

Comando:

```bash
python3 src/subgroup_gap_analysis.py --config config/config.yaml
```

Este script evalúa si el modelo se comporta de forma homogénea entre subgrupos operativos. Los grupos pueden definirse por proyecto, tipo de incidencia, prioridad, periodo temporal o presencia de descripción.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `subgroup_gap_metrics.csv` | `reports/` | Métricas detalladas por subgrupo |
| `subgroup_gap_summary.csv` | `reports/` | Resumen de gaps máximos y medios |

Estos archivos permiten detectar degradación diferencial y diferencias de calidad predictiva que no aparecen en métricas globales.

### Degradación controlada de datos

Comando:

```bash
python3 src/degradation_experiments.py --config config/config.yaml
```

Este script simula problemas de calidad de datos. Entre los escenarios evaluados se incluyen pérdida de descripción, ruido en prioridad, pérdida de valores categóricos y submuestreo de la clase positiva.

Salida habitual:

```text
reports/degradation_metrics.csv
```

El resultado permite estudiar qué ocurre cuando la calidad del dato se deteriora de forma controlada. Esta fase es clave para demostrar que no todos los problemas de calidad afectan igual al modelo.

### Limpieza de datos

Comando:

```bash
python3 src/data_cleaning_experiment.py --config config/config.yaml
```

Este script aplica reglas de limpieza y compara el dataset original preparado con una versión limpiada. El objetivo no es asumir que limpiar siempre mejora, sino medir empíricamente qué métricas mejoran y cuáles empeoran.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `jira_issues_cleaned.csv` | `data/processed/` | Dataset limpiado |
| `data_cleaning_quality.csv` | `reports/` | Comparación de calidad |
| `data_cleaning_model_metrics.csv` | `reports/` | Comparación de rendimiento |

Este experimento ayuda a defender que la limpieza debe evaluarse con varias métricas, no solo con F1-score o accuracy.

### Drift temporal

Comando:

```bash
python3 src/temporal_drift_experiment.py --config config/config.yaml
```

Este script compara una partición aleatoria con una partición temporal. También analiza cambios en la distribución de variables entre entrenamiento y test temporal.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `temporal_drift_metrics.csv` | `reports/` | Comparación entre split aleatorio y temporal |
| `temporal_numeric_shift.csv` | `reports/` | Cambios en variables numéricas |
| `temporal_categorical_shift.csv` | `reports/` | Cambios en variables categóricas |
| `temporal_period_summary.csv` | `reports/` | Resumen de periodos |

Esta fase permite comprobar si la evaluación aleatoria es optimista frente a una evaluación más cercana a uso futuro.

### Reentrenamiento

Comando:

```bash
python3 src/retraining_experiment.py --config config/config.yaml
```

El script compara estrategias de mantenimiento del modelo: modelo estático, reentrenamiento acumulado y ventana deslizante. La finalidad es comprobar si actualizar el modelo mejora realmente el comportamiento o si introduce nuevos trade-offs.

Salida habitual:

```text
reports/retraining_metrics.csv
```

Estos resultados permiten discutir que el reentrenamiento no debe aplicarse automáticamente. En un entorno real, un modelo reentrenado debería compararse con el modelo vigente antes de sustituirlo.

### Estabilidad entre semillas

Comando:

```bash
python3 src/stability_experiment.py --config config/config.yaml --seeds 10
```

Este script repite el entrenamiento y la evaluación con distintas semillas aleatorias. Su objetivo es analizar si los resultados son estables o dependen excesivamente de una partición concreta.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `stability_seed_metrics.csv` | `reports/` | Métricas por semilla |
| `stability_seed_summary.csv` | `reports/` | Media, desviación estándar y rango |

Esta fase permite interpretar los resultados con mayor cautela cuando existe variabilidad entre ejecuciones.

### Posibles mundos e incertidumbre de datos

Comando:

```bash
python3 src/possible_worlds_sensitivity.py --config config/config.yaml --worlds 20
```

Este script genera versiones alternativas plausibles de determinados registros para analizar si pequeñas variaciones en datos inciertos modifican las predicciones. Cada versión alternativa se interpreta como un posible mundo.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `possible_worlds_predictions.csv` | `reports/` | Predicciones por mundo |
| `possible_worlds_sensitivity_summary.csv` | `reports/` | Sensibilidad por observación |
| `possible_worlds_sensitivity_aggregate.csv` | `reports/` | Resumen agregado |

Aunque en la configuración aplicada no se observó variabilidad relevante, el experimento se conserva como análisis de sensibilidad metodológica.

### Auditoría de sesgos y fairness operacional

Comando:

```bash
python3 src/bias_fairness_audit.py --config config/config.yaml
```

Este script amplía la evaluación desde una perspectiva de sesgos operacionales. No analiza discriminación demográfica directa, porque el dataset público está anonimizado y no contiene atributos sensibles clásicos. En su lugar, evalúa diferencias de rendimiento entre subgrupos operativos.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `bias_fairness_global_metrics.csv` | `reports/` | Métricas globales de algoritmos evaluados |
| `bias_fairness_group_metrics.csv` | `reports/` | Métricas por subgrupo |
| `bias_fairness_disparity_summary.csv` | `reports/` | Resumen de brechas entre grupos |

Esta fase permite justificar la incorporación de fairness operacional, degradación diferencial y métricas de diferencia entre grupos.

### Escenarios controlados de sesgo

Comando:

```bash
python3 src/bias_scenarios_experiment.py --config config/config.yaml
```

Este script simula fuentes de sesgo como infrarrepresentación, missingness diferenciado, sesgo histórico o sesgo de evaluación. El objetivo es observar si el modelo es sensible a perturbaciones que afectan de forma distinta a determinados grupos.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `bias_scenario_metrics.csv` | `reports/` | Métricas globales por escenario |
| `bias_scenario_disparities.csv` | `reports/` | Brechas entre grupos por escenario |

Estos resultados refuerzan la discusión sobre fuentes de sesgo en datos y modelos.

### Mitigación de sesgos

Comando:

```bash
python3 src/bias_mitigation_experiment.py --config config/config.yaml
```

Este script compara estrategias de mitigación como pesos de clase, reponderación por grupo, combinación de pesos, ajuste global de umbral y ajuste orientado a grupos.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `bias_mitigation_metrics.csv` | `reports/` | Métricas globales por estrategia |
| `bias_mitigation_group_metrics.csv` | `reports/` | Métricas por subgrupo tras mitigación |
| `bias_mitigation_disparities.csv` | `reports/` | Brechas por estrategia |
| `bias_mitigation_tradeoffs.csv` | `reports/` | Beneficios y costes frente al baseline |

Esta fase permite analizar si una técnica reduce determinadas brechas y qué coste introduce en rendimiento, calibración, precision o recall.

### Proxies de riesgo en gestión de proyectos

Comando:

```bash
python3 src/project_risk_proxy_export.py --config config/config.yaml
```

Este script conecta los resultados técnicos con la gestión de proyectos. La variable `late_resolution` se interpreta como proxy de riesgo de plazo, mientras que la duración observada se utiliza como aproximación indirecta al esfuerzo. La dimensión de alcance queda más limitada por la ausencia de campos como story points, sprint, épicas o componentes funcionales completos.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `project_risk_proxy_summary.csv` | `reports/` | Resumen global de proxies |
| `project_risk_proxy_by_group.csv` | `reports/` | Proxies por subgrupo |
| `project_risk_proxy_dataset.csv` | `reports/` | Dataset enriquecido con proxies |

Estos archivos permiten relacionar el modelo con riesgos de plazo, esfuerzo y alcance de forma exploratoria.

### Exportación de KPIs para Power BI

Comando:

```bash
python3 src/monitoring_kpis_export.py --config config/config.yaml
```

Este script no entrena modelos nuevos. Su función es reorganizar los resultados generados por el pipeline en tablas adecuadas para visualización, comparación y monitorización en Power BI.

Salidas habituales:

| Archivo | Ubicación | Uso en Power BI |
|---|---|---|
| `performance_kpis_long.csv` | `data/powerbi/` | Tabla principal de KPIs en formato largo |
| `kpi_catalog.csv` | `data/powerbi/` | Catálogo de métricas y definiciones |
| `kpi_snapshot.csv` | `data/powerbi/` | Tarjetas resumen |
| `data_quality_summary.csv` | `data/powerbi/` | Panel de calidad |
| `category_distributions.csv` | `data/powerbi/` | Panel de composición |
| `calibration_bins.csv` | `data/powerbi/` | Visualizaciones de calibración |

Estas salidas permiten construir paneles sobre calidad, rendimiento, calibración, drift, subgrupos, fairness, mitigación, proxies de riesgo y auditoría.

### Auditoría y trazabilidad

Comando:

```bash
python3 src/audit_manifest.py --config config/config.yaml
```

Este script genera un manifiesto de auditoría con inventario de archivos, dependencias, fechas de modificación y hashes. Su finalidad es permitir que el pipeline sea revisable y reproducible.

Salidas habituales:

| Archivo | Ubicación | Uso en el TFM |
|---|---|---|
| `audit_manifest.json` | `reports/` | Manifiesto completo de ejecución |
| `audit_file_inventory.csv` | `reports/` | Inventario de archivos y hashes |
| `audit_runbook.md` | `reports/` | Guía de ejecución y trazabilidad |

Esta fase materializa la dimensión de auditoría del TFM. Los hashes permiten detectar cambios en scripts o resultados, y el runbook facilita reconstruir la ejecución.

## Comprobación de resultados generados

Tras ejecutar el pipeline completo, se recomienda comprobar que existen las carpetas y salidas principales:

```bash
ls data/raw
ls data/processed
ls reports
ls data/powerbi
```

También puede verificarse la existencia de archivos clave:

```bash
ls data/raw/jira_issues.csv
ls data/processed/jira_issues_prepared.csv
ls reports/model_metrics.csv
ls reports/advanced_model_metrics.csv
ls reports/degradation_metrics.csv
ls reports/retraining_metrics.csv
ls reports/bias_fairness_global_metrics.csv
ls reports/bias_mitigation_tradeoffs.csv
ls reports/project_risk_proxy_summary.csv
ls data/powerbi/performance_kpis_long.csv
ls reports/audit_file_inventory.csv
```

Si alguno de estos archivos no existe, debe revisarse el script correspondiente y comprobar que se ha ejecutado en el orden adecuado. También debe revisarse que `config/config.yaml` contenga las rutas correctas y que el entorno virtual esté activado.


## Relación entre scripts y resultados

| Script | Resultado generado | 
|---|---|
| `extract_from_mongodb.py` | CSV inicial desde MongoDB |
| `prepare_dataset.py` | Dataset analítico preparado |
| `eda_quality.py` | Calidad inicial y distribuciones |
| `train_baseline.py` | Métricas baseline |
| `threshold_tuning.py` | Umbrales y trade-off precision/recall |
| `advanced_metrics.py` | Métricas avanzadas y calibración |
| `subgroup_gap_analysis.py` | Gaps por subgrupo |
| `degradation_experiments.py` | Sensibilidad ante degradación |
| `data_cleaning_experiment.py` | Comparación antes/después de limpieza |
| `temporal_drift_experiment.py` | Split aleatorio frente a temporal |
| `retraining_experiment.py` | Estrategias de reentrenamiento |
| `stability_experiment.py` | Variabilidad entre semillas |
| `possible_worlds_sensitivity.py` | Sensibilidad ante incertidumbre |
| `bias_fairness_audit.py` | Fairness operacional y disparidades |
| `bias_scenarios_experiment.py` | Escenarios controlados de sesgo |
| `bias_mitigation_experiment.py` | Mitigación y trade-offs |
| `project_risk_proxy_export.py` | Proxies de riesgo de proyecto |
| `monitoring_kpis_export.py` | CSV para Power BI |
| `audit_manifest.py` | Inventario, hashes y runbook |

Como muestra la tabla, los resultados incluidos en el TFM no proceden de cálculos manuales aislados, sino de una secuencia reproducible de scripts. 

