

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys, os, warnings
warnings.filterwarnings('ignore')

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (silhouette_score, davies_bouldin_score,
                              calinski_harabasz_score)
from sklearn.neighbors import NearestNeighbors
from scipy.cluster.hierarchy import dendrogram, linkage

# ═══════════════════════════════════════════════════════════
#  CONFIGURACIÓN – CAMBIA ESTOS VALORES
# ═══════════════════════════════════════════════════════════
CSV_PATH      = "Mall_Customers.csv"  

# Nombres de columnas (tal como aparecen en tu CSV)
COL_ID        = "CustomerID"           # columna ID (se descarta)
COL_GENDER    = "Genre"                # columna género
COL_AGE       = "Age"                  # columna edad
COL_INCOME    = "Annual Income (k$)"   # columna ingreso anual
COL_SPENDING  = "Spending Score (1-100)"  # columna spending score

# Número de clusters para K-Means y Jerárquico
# (si pones None el script lo elige automáticamente)
N_CLUSTERS    = None
# ═══════════════════════════════════════════════════════════

COLORS = ['#e74c3c','#3498db','#2ecc71','#f39c12','#9b59b6',
          '#1abc9c','#e67e22','#34495e','#e91e8c','#00bcd4']

print("=" * 60)
print("EJERCICIO 4: AGRUPAMIENTO DE CLIENTES – MALL CUSTOMERS")
print("=" * 60)

# ── CARGA DEL CSV ────────────────────────────────────────────
if not os.path.exists(CSV_PATH):
    print(f"\n❌ ERROR: No se encontró '{CSV_PATH}'")
    print(f"   Carpeta actual: {os.getcwd()}")
    sys.exit(1)

print(f"\n📂 Cargando: {CSV_PATH}")
df = pd.read_csv(CSV_PATH)
print(f"   Filas: {len(df):,} | Columnas: {list(df.columns)}")

# Validar columnas
required = [COL_AGE, COL_INCOME, COL_SPENDING]
for col in required:
    if col not in df.columns:
        print(f"\n❌ ERROR: Columna '{col}' no encontrada.")
        print(f"   Columnas disponibles: {list(df.columns)}")
        print("   Edita las variables COL_* al inicio del script.")
        sys.exit(1)

# Manejo de género (opcional)
has_gender = COL_GENDER in df.columns
if has_gender:
    le = LabelEncoder()
    df['Genre_enc'] = le.fit_transform(df[COL_GENDER].fillna('Unknown'))

df = df.dropna(subset=required)
print(f"   Filas sin nulos: {len(df):,}")
print(f"\nEstadísticas:\n{df[required].describe().round(2)}")

# Features para clustering
features = [COL_AGE, COL_INCOME, COL_SPENDING]
X = df[features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ── EDA ──────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Ejercicio 4: EDA – Mall Customers', fontsize=14, fontweight='bold')

ax = axes[0, 0]
if has_gender:
    counts = df[COL_GENDER].value_counts()
    ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%',
           colors=['#3498db','#e91e8c'], startangle=90,
           wedgeprops=dict(edgecolor='white', linewidth=2))
    ax.set_title('Distribución por Género', fontweight='bold')
else:
    ax.hist(df[COL_AGE], bins=20, color='#9b59b6', edgecolor='black', alpha=0.8)
    ax.set_title('Distribución de Edad', fontweight='bold')

ax = axes[0, 1]
ax.hist(df[COL_AGE], bins=20, color='#9b59b6', edgecolor='black', alpha=0.8)
ax.axvline(df[COL_AGE].mean(), color='red', linestyle='--', lw=2,
           label=f'Media={df[COL_AGE].mean():.1f}')
ax.set_title('Distribución de Edad', fontweight='bold'); ax.legend()

ax = axes[0, 2]
sc = ax.scatter(df[COL_INCOME], df[COL_SPENDING],
                c=df[COL_AGE], cmap='viridis', s=60, alpha=0.7, edgecolors='none')
plt.colorbar(sc, ax=ax, label='Edad')
ax.set_title('Ingreso vs Spending Score\n(color=Edad)', fontweight='bold')
ax.set_xlabel(COL_INCOME); ax.set_ylabel(COL_SPENDING)

ax = axes[1, 0]
ax.hist(df[COL_INCOME], bins=20, color='#3498db', edgecolor='black', alpha=0.8)
ax.axvline(df[COL_INCOME].mean(), color='red', linestyle='--', lw=2)
ax.set_title('Distribución de Ingreso Anual', fontweight='bold')

ax = axes[1, 1]
corr = df[features].corr()
sns.heatmap(corr, annot=True, fmt='.3f', cmap='RdYlGn', center=0, ax=ax,
            linewidths=0.5, annot_kws={'size':10,'weight':'bold'})
ax.set_title('Matriz de Correlación', fontweight='bold')

ax = axes[1, 2]
ax.hist(df[COL_SPENDING], bins=20, color='#2ecc71', edgecolor='black', alpha=0.8)
ax.axvline(df[COL_SPENDING].mean(), color='red', linestyle='--', lw=2)
ax.set_title('Distribución Spending Score', fontweight='bold')

plt.tight_layout()
plt.savefig('ej4_eda.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ej4_eda.png guardada")

# ── BÚSQUEDA DE K ÓPTIMO ─────────────────────────────────────
print("\n🔍 Buscando K óptimo para K-Means...")
k_range = range(2, 11)
inertias, silhouettes, dbs, chs = [], [], [], []
for k in k_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=20, random_state=42)
    lbl = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    silhouettes.append(silhouette_score(X_scaled, lbl))
    dbs.append(davies_bouldin_score(X_scaled, lbl))
    chs.append(calinski_harabasz_score(X_scaled, lbl))

# Elegir K automáticamente si no se especificó
if N_CLUSTERS is None:
    K_OPT = list(k_range)[np.argmax(silhouettes)]
    print(f"   K óptimo elegido automáticamente: {K_OPT} (mejor Silhouette Score)")
else:
    K_OPT = N_CLUSTERS
    print(f"   K configurado manualmente: {K_OPT}")

fig, axes = plt.subplots(1, 4, figsize=(18, 4))
fig.suptitle('Determinación del K Óptimo – K-Means', fontsize=13, fontweight='bold')
for ax, title, data, color in zip(axes,
        ['Elbow (Inercia)', 'Silhouette (↑)', 'Davies-Bouldin (↓)', 'Calinski-Harabasz (↑)'],
        [inertias, silhouettes, dbs, chs],
        ['#e74c3c','#2ecc71','#e67e22','#3498db']):
    ax.plot(list(k_range), data, 'o-', color=color, lw=2, ms=8)
    ax.axvline(K_OPT, color='black', linestyle='--', alpha=0.6, label=f'k={K_OPT}')
    idx = list(k_range).index(K_OPT)
    ax.scatter([K_OPT], [data[idx]], s=180, color='black', zorder=5)
    ax.set_xlabel('k'); ax.set_title(title, fontweight='bold')
    ax.set_xticks(list(k_range)); ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig('ej4_k_optimo.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ej4_k_optimo.png guardada")

# ── MODELO 1: K-MEANS ────────────────────────────────────────
print(f"\n🔵 K-Means (k={K_OPT})...")
kmeans = KMeans(n_clusters=K_OPT, init='k-means++', n_init=50, random_state=42)
df['Cluster_KMeans'] = kmeans.fit_predict(X_scaled)
sil_km = silhouette_score(X_scaled, df['Cluster_KMeans'])
db_km  = davies_bouldin_score(X_scaled, df['Cluster_KMeans'])
ch_km  = calinski_harabasz_score(X_scaled, df['Cluster_KMeans'])
print(f"   Silhouette={sil_km:.4f} | Davies-Bouldin={db_km:.4f} | CH={ch_km:.1f}")

print("\n📊 Perfil de clusters:")
perfil = df.groupby('Cluster_KMeans')[features].mean().round(1)
perfil['Tamaño'] = df['Cluster_KMeans'].value_counts().sort_index()
print(perfil)

# ── MODELO 2: DBSCAN ─────────────────────────────────────────
print("\n🟠 DBSCAN (selección automática de eps)...")
nbrs = NearestNeighbors(n_neighbors=5).fit(X_scaled)
dists, _ = nbrs.kneighbors(X_scaled)
dists_sorted = np.sort(dists[:, 4])
# Elegir eps en el codo (percentil 90)
eps_auto = float(np.percentile(dists_sorted, 90))
print(f"   eps elegido automáticamente: {eps_auto:.3f}")

fig_e, ax_e = plt.subplots(figsize=(8, 4))
ax_e.plot(dists_sorted, color='#e74c3c', lw=2)
ax_e.axhline(eps_auto, color='blue', linestyle='--', lw=2, label=f'eps={eps_auto:.3f}')
ax_e.set_title('Curva k-NN para selección de eps (DBSCAN)', fontweight='bold')
ax_e.set_xlabel('Puntos'); ax_e.set_ylabel('5-NN Distance'); ax_e.legend()
plt.tight_layout()
plt.savefig('ej4_dbscan_eps.png', dpi=150, bbox_inches='tight')
plt.show()

dbscan = DBSCAN(eps=eps_auto, min_samples=5)
df['Cluster_DBSCAN'] = dbscan.fit_predict(X_scaled)
n_clust_db = len(set(df['Cluster_DBSCAN'])) - (1 if -1 in df['Cluster_DBSCAN'].values else 0)
n_noise    = (df['Cluster_DBSCAN'] == -1).sum()
print(f"   Clusters: {n_clust_db} | Ruido: {n_noise} pts ({n_noise/len(df)*100:.1f}%)")

mask_v = df['Cluster_DBSCAN'] != -1
if n_clust_db > 1 and mask_v.sum() > 1 and len(set(df.loc[mask_v,'Cluster_DBSCAN'])) > 1:
    sil_db = silhouette_score(X_scaled[mask_v], df.loc[mask_v,'Cluster_DBSCAN'])
    print(f"   Silhouette (sin ruido): {sil_db:.4f}")
else:
    sil_db = 0.0

# ── MODELO 3: JERÁRQUICO ─────────────────────────────────────
print(f"\n🟢 Clustering Jerárquico (Ward, k={K_OPT})...")
lm = linkage(X_scaled, method='ward')
fig_d, ax_d = plt.subplots(figsize=(14, 5))
dendrogram(lm, ax=ax_d, truncate_mode='lastp', p=30, leaf_rotation=90,
           leaf_font_size=9, show_contracted=True)
ax_d.set_title('Dendrograma – Clustering Jerárquico (Ward)', fontsize=13, fontweight='bold')
ax_d.set_xlabel('Clientes'); ax_d.set_ylabel('Distancia')
plt.tight_layout()
plt.savefig('ej4_dendrograma.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ej4_dendrograma.png guardada")

hier = AgglomerativeClustering(n_clusters=K_OPT, linkage='ward')
df['Cluster_Hier'] = hier.fit_predict(X_scaled)
sil_hier = silhouette_score(X_scaled, df['Cluster_Hier'])
db_hier  = davies_bouldin_score(X_scaled, df['Cluster_Hier'])
print(f"   Silhouette={sil_hier:.4f} | Davies-Bouldin={db_hier:.4f}")

# ── PCA + VISUALIZACIÓN ──────────────────────────────────────
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)
var = pca.explained_variance_ratio_
print(f"\n📐 PCA: PC1={var[0]*100:.1f}% | PC2={var[1]*100:.1f}% | Total={sum(var)*100:.1f}%")

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Comparación de Modelos de Clustering (PCA 2D)', fontsize=14, fontweight='bold')

for ax, col, title in zip(axes,
        ['Cluster_KMeans','Cluster_DBSCAN','Cluster_Hier'],
        [f'K-Means (k={K_OPT})\nSil={sil_km:.3f}',
         f'DBSCAN\nClusters={n_clust_db} | Ruido={n_noise}',
         f'Jerárquico (Ward)\nSil={sil_hier:.3f}']):
    for lbl in sorted(df[col].unique()):
        mask = df[col] == lbl
        color = '#aaaaaa' if lbl == -1 else COLORS[lbl % len(COLORS)]
        label = 'Ruido' if lbl == -1 else f'C{lbl}'
        ax.scatter(X_pca[mask,0], X_pca[mask,1], c=color, label=label,
                   s=50, alpha=0.8, edgecolors='none')
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel(f'PC1 ({var[0]*100:.1f}%)'); ax.set_ylabel(f'PC2 ({var[1]*100:.1f}%)')
    ax.legend(fontsize=8, markerscale=1.5)

plt.tight_layout()
plt.savefig('ej4_pca_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ej4_pca_clusters.png guardada")

# ── PERFIL DETALLADO K-MEANS ─────────────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(16, 10))
fig.suptitle('Análisis Detallado de Segmentos K-Means', fontsize=13, fontweight='bold')
c_colors = [COLORS[i] for i in range(K_OPT)]

ax = axes[0, 0]
centroids_orig = scaler.inverse_transform(kmeans.cluster_centers_)
for k in range(K_OPT):
    mask = df['Cluster_KMeans'] == k
    ax.scatter(df.loc[mask, COL_INCOME], df.loc[mask, COL_SPENDING],
               c=COLORS[k], s=60, alpha=0.8, label=f'C{k}', edgecolors='none')
ax.scatter(centroids_orig[:,1], centroids_orig[:,2],
           c='black', marker='*', s=300, zorder=5, label='Centroide')
ax.set_xlabel(COL_INCOME); ax.set_ylabel(COL_SPENDING)
ax.set_title('Ingreso vs Spending por Cluster', fontweight='bold'); ax.legend(fontsize=8)

ax = axes[0, 1]
for k in range(K_OPT):
    mask = df['Cluster_KMeans'] == k
    ax.scatter(df.loc[mask, COL_AGE], df.loc[mask, COL_SPENDING],
               c=COLORS[k], s=60, alpha=0.8, label=f'C{k}', edgecolors='none')
ax.set_xlabel(COL_AGE); ax.set_ylabel(COL_SPENDING)
ax.set_title('Edad vs Spending por Cluster', fontweight='bold'); ax.legend(fontsize=8)

ax = axes[0, 2]
sizes = df['Cluster_KMeans'].value_counts().sort_index()
ax.bar([f'C{i}' for i in sizes.index], sizes.values,
       color=c_colors, edgecolor='black', lw=0.7)
ax.set_title('Tamaño de Cada Cluster', fontweight='bold'); ax.set_ylabel('Clientes')
for bar, val in zip(ax.patches, sizes.values):
    ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
            str(val), ha='center', fontweight='bold')

for subplot_idx, (feat, ax) in enumerate(
        zip(features, [axes[1,0], axes[1,1], axes[1,2]])):
    data_bp = [df[df['Cluster_KMeans']==k][feat].values for k in range(K_OPT)]
    bp = ax.boxplot(data_bp, patch_artist=True,
                    medianprops=dict(color='black', lw=2))
    for patch, color in zip(bp['boxes'], c_colors):
        patch.set_facecolor(color); patch.set_alpha(0.7)
    ax.set_xticklabels([f'C{k}' for k in range(K_OPT)])
    ax.set_title(f'"{feat}"\npor Cluster', fontweight='bold'); ax.set_ylabel(feat)

plt.tight_layout()
plt.savefig('ej4_segmentos_detalle.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ ej4_segmentos_detalle.png guardada")

# ── TABLA RESUMEN ────────────────────────────────────────────
print("\n" + "="*60)
print("RESUMEN FINAL DE MÉTRICAS")
print("="*60)
summary = pd.DataFrame({
    'Modelo':          ['K-Means', 'DBSCAN', 'Jerárquico'],
    'N Clusters':      [K_OPT, n_clust_db, K_OPT],
    'Silhouette':      [sil_km, sil_db, sil_hier],
    'Davies-Bouldin':  [db_km, 0.0, db_hier],
    'Puntos Ruido':    [0, n_noise, 0],
})
print(summary.round(4).to_string(index=False))

print(f"\n📊 Perfil final de segmentos K-Means:")
perfil_final = df.groupby('Cluster_KMeans').agg(
    Edad=('Age', 'mean'),
    Ingreso=(COL_INCOME, 'mean'),
    Spending=(COL_SPENDING, 'mean'),
    Tamaño=(features[0], 'count')
).round(1)
print(perfil_final)
print("\n✅ Análisis completo. Todos los gráficos guardados.")