# -*- coding: utf-8 -*-
"""
Análise do Dataset de Indicadores de Saúde para Diabetes - Parte 2 (Execução Final)

Este script executa a versão final da análise de clusterização, focando apenas
no melhor algoritmo (K-Means) com o melhor parâmetro (k=2) já descoberto.
Etapas:
1.  Carrega e prepara os dados (com amostragem).
2.  Aplica a Seleção de Features com Random Forest.
3.  Executa o K-Means diretamente com k=2.
4.  Gera as métricas de avaliação, a visualização PCA e a análise cruzada.
"""

# --------------------------------------------------------------------------
# SEÇÃO 1: IMPORTAÇÃO DAS BIBLIOTECAS
# --------------------------------------------------------------------------
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Módulos do Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------------------------------
# SEÇÃO 2: CARREGAMENTO E PREPARAÇÃO DOS DADOS
# --------------------------------------------------------------------------

try:
    df_full = pd.read_csv('diabetes_binary_health_indicators_BRFSS2015.csv')
    print("Dataset completo carregado com sucesso!")
except FileNotFoundError:
    print("Erro: Arquivo 'diabetes_binary_health_indicators_BRFSS2015.csv' não encontrado.")
    exit()

# Usando uma amostra para garantir a execução rápida.
# Se quiser rodar no dataset completo, apenas comente ou remova as duas linhas abaixo.
SAMPLE_SIZE = 220000
df = df_full.sample(n=SAMPLE_SIZE, random_state=42)
print(f"\nTrabalhando com uma amostra de {SAMPLE_SIZE} pontos.")

target_column = 'Diabetes_binary'
X = df.drop(target_column, axis=1)
y = df[target_column]

# --------------------------------------------------------------------------
# SEÇÃO 3: SELEÇÃO DE FEATURES COM RANDOM FOREST
# --------------------------------------------------------------------------
print("\n--- Selecionando as Features Mais Importantes ---")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X, y)

feature_importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)
N_TOP_FEATURES = 10
top_features = feature_importances.head(N_TOP_FEATURES).index

X_selecionado = X[top_features]

print("\nPadronizando as features selecionadas...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_selecionado)

# --------------------------------------------------------------------------
# SEÇÃO 4: EXECUÇÃO E AVALIAÇÃO DO K-MEANS FINAL
# --------------------------------------------------------------------------

# Define o k ótimo que já descobrimos na análise anterior
k_otimo = 2 
print(f"\nExecutando o K-Means final com k={k_otimo}...")

kmeans_final = KMeans(n_clusters=k_otimo, init='k-means++', random_state=42, n_init='auto')
kmeans_labels = kmeans_final.fit_predict(X_scaled)

print("Cálculo das métricas de avaliação concluído.")

# Calcula e exibe as métricas para o K-Means
sil_score = silhouette_score(X_scaled, kmeans_labels)
calinski_score = calinski_harabasz_score(X_scaled, kmeans_labels)
davies_score = davies_bouldin_score(X_scaled, kmeans_labels)

print("\n--- Métricas de Clusterização para K-Means ---")
print(f"Silhouette Score (↑): {sil_score:.4f}")
print(f"Calinski-Harabasz (↑): {calinski_score:.2f}")
print(f"Davies-Bouldin (↓): {davies_score:.4f}")

# --- Visualização do Cluster com PCA ---
print("\nReduzindo dimensionalidade com PCA para visualização...")
pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=kmeans_labels, palette='viridis', s=20, alpha=0.7)
plt.title(f'Visualização dos Clusters do K-Means (k={k_otimo})')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend(title='Clusters')
plt.show()

# --- Análise Cruzada com os Rótulos Verdadeiros ---
print("\n--- Análise Cruzada: Rótulos Verdadeiros vs. Clusters do K-Means ---")
df_analise = pd.DataFrame({'Rótulo Verdadeiro': y.map({0: 'Não Diabético', 1: 'Diabético'}), 'Cluster K-Means': kmeans_labels})
tabela_cruzada = pd.crosstab(df_analise['Cluster K-Means'], df_analise['Rótulo Verdadeiro'])
print(tabela_cruzada)