# 📊 Evento Evaluativo 4 – Analisis de Datos

**Asignatura:** Analisis de Datos  
**Docente:** Daniel Alexis Nieto Mora  
**Grupo:** 190304018-1  
**Periodo:** 2026-2  

## Integrantes
Juan David Palacio
Sebastian Cuencar
Juan José Rúa
Julián Ramírez

## Contenido
- Ejercicio 3: clasificacion de reseñas (notebooks/ejercicio3_Amazon_Reviews.ipynb)
- Ejercicio 4: clustering de clientes (ejercicio4_mall_customers.ipynb)

## Estructura de datos
- datasets/Ejercicio 3/train.ft.txt.bz2
- datasets/Ejercicio 3/test.ft.txt.bz2
- datasets/Ejercicio 4/Mall_Customers.csv

## Requisitos
```bash
pip install -r requirements.txt
python -c "import nltk; nltk.download('stopwords')"
```

## Ejercicio 3: Analisis de sentimientos (Amazon Reviews)
**Objetivo:** clasificar reseñas como positivas o negativas a partir del texto.

**Proceso:**
- Limpieza: minusculas, regex, stopwords y stemming (PorterStemmer).
- Vectorizacion: TF-IDF con unigramas y bigramas.
- Modelos: MultinomialNB, Logistic Regression, Linear SVM.

**Metricas:** accuracy, F1 macro, F1 weighted, matriz de confusion.

**Visualizaciones:** balance de clases, matrices de confusion, t-SNE (muestra).

## Ejercicio 4: Agrupamiento de clientes (Mall Customers)
**Objetivo:** segmentar clientes segun edad, ingreso y spending score.

**Proceso:**
- EDA: distribuciones, genero, dispersion y correlacion.
- Preprocesamiento: estandarizacion y opcion de incluir genero.
- Modelos: K-Means, DBSCAN, Jerarquico (Ward).
- PCA 2D para comparar visualmente los clusters.

**Metricas:** Silhouette, Davies-Bouldin, Calinski-Harabasz.

## Ejecucion
1. Abre los notebooks con Jupyter o VS Code.
2. Ejecuta las celdas en orden. Los graficos se muestran en el notebook (no se guardan).

## Notas
- El notebook principal del ejercicio 4 esta en la raiz.
- Si quieres acelerar el ejercicio 3, puedes activar muestreo en la celda de carga.