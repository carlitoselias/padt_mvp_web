# PADT MVP

Este repositorio contiene notebooks y utilidades para procesar registros de telefonía
móvil y generar indicadores de movilidad. Los cuadernos se apoyan en PySpark y
requieren configurar rutas locales a los datos.

## Preparación del entorno

1. Instalar Python 3.
2. Instalar las dependencias del proyecto:

```bash
pip install -r requirements.txt
```

3. Opcionalmente se puede instalar el paquete local para habilitar las utilidades
   en `src/`:

```bash
pip install -e .
```

4. Ajustar las rutas en `config/config.yaml` para apuntar a los directorios de
   datos y las preferencias de ejecución de Spark.

## Ejecución de notebooks y scripts

Los notebooks numerados (`00_catalogo_antenas.ipynb`, `01_creacion_viajes.ipynb`,
`02_seleccion_antenas.ipynb`, etc.) muestran el flujo de trabajo para procesar
los datos paso a paso. Puede abrirlos con JupyterLab o Jupyter Notebook.

## Utilidades de Spark

El módulo `src/mtt/spark_utils.py` proporciona la función `create_spark_session`.
Esta lee el archivo `config/config.yaml` y construye una `SparkSession` con los
parámetros definidos, asegurando que los notebooks utilicen la misma
configuración de Spark.

Solo se debe importar esta celda en un .ipynb para iniciar la sesión de spark.

```
from mtt.spark_utils import create_spark_session
# Carga parámetros de config/config.yaml
spark = create_spark_session("config/config.yaml")
sc  = spark.sparkContext
sql = spark.sql
```