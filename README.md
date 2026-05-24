# 📊 Evento Evaluativo 4 – Análisis de Datos

**Asignatura:** Análisis de Datos  
**Docente:** Daniel Alexis Nieto Mora  
**Grupo:** 190304018-1  
**Periodo:** 2025-2  


## 🔵 Ejercicio 3: Análisis de Sentimientos en Reseñas Amazon

### Descripción
Modelo de clasificación supervisada para predecir si una reseña de producto
es **positiva** o **negativa** a partir de texto.

### Dataset
- **Fuente:** [Amazon Reviews Dataset – Kaggle](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
- **Variables:** texto de la reseña, etiqueta de sentimiento
- **Preprocesamiento:** minúsculas, eliminación de puntuación, stopwords,
  lematización (NLTK), vectorización TF-IDF (unigramas + bigramas)

### Modelos Aplicados

| Modelo | Justificación |
|--------|--------------|
| **Complement Naive Bayes** | Especialmente efectivo en clasificación de texto desbalanceado; muy rápido |
| **Regresión Logística** | Alta interpretabilidad mediante coeficientes; excelente con TF-IDF |
| **SVM Lineal (LinearSVC)** | Estado del arte en clasificación de texto de alta dimensionalidad |

### Métricas Usadas
- Accuracy, F1-Score ponderado, Precision, Recall
- Matriz de Confusión
- Cross-Validation estratificado (5-fold)

### Visualizaciones
- Distribuciones de clases y longitud de reseñas
- WordCloud de palabras frecuentes por sentimiento
- t-SNE (SVD 50D → 2D) para visualización de agrupaciones
- Palabras más influyentes por coeficiente (Logistic Regression)

---

## 🟢 Ejercicio 4: Agrupamiento de Clientes – Mall Customers

### Descripción
Segmentación **no supervisada** de clientes de un mall comercial
según variables demográficas y de comportamiento de compra.

### Dataset
- **Fuente:** [Mall Customers Dataset – Kaggle](https://www.kaggle.com/datasets/vjchoudhary7/customer-segmentation-tutorial-in-python)
- **Variables:** Género, Edad, Ingreso Anual (k$), Spending Score (1–100)

### Modelos Aplicados

| Modelo | Justificación |
|--------|--------------|
| **K-Means (k=5)** | Rápido, escalable, identifica clusters compactos; k elegido por codo + Silhouette |
| **DBSCAN** | Detecta clusters de forma arbitraria y outliers sin definir k a priori |
| **Clustering Jerárquico (Ward)** | Permite visualizar estructura del dendrograma; confirma k=5 |

### Métricas Usadas
- **Silhouette Score:** cohesión vs separación de clusters (↑ mejor)
- **Davies-Bouldin Score:** compacidad y separación (↓ mejor)
- **Calinski-Harabasz Score:** ratio de dispersión inter/intra cluster (↑ mejor)

### Visualizaciones
- EDA: distribuciones, correlación, boxplots por género
- Método del codo + Silhouette para K óptimo
- Dendrograma del clustering jerárquico
- PCA 2D comparando los 3 modelos
- Perfil detallado de cada segmento K-Means

### Segmentos Identificados (K-Means k=5)
| Segmento | Ingreso | Spending | Estrategia |
|----------|---------|----------|------------|
|  Premium | Alto | Alto | Fidelización VIP |
|  Impulsivo | Bajo | Alto | Monitoreo crediticio |
|  Conservador | Alto | Bajo | Promociones dirigidas |
|  Económico | Bajo | Bajo | Descuentos masivos |
|  Moderado | Medio | Medio | Campañas generales |

---

## ⚙️ Instalación

```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk wordcloud scipy
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## ▶️ Ejecución

```bash
python ejercicio4_clustering.py
```
