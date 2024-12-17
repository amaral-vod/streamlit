import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Charger le jeu de données Iris
iris = sns.load_dataset("iris")

# Titre de l'application
st.title("Exploration et Analyse du Jeu de Données Iris")

# Afficher un résumé général
st.write("### Aperçu des données")
st.dataframe(iris)

# Distribution des caractéristiques
st.write("### Distribution des caractéristiques par espèce")
fig1, ax1 = plt.subplots(figsize=(10, 6))
sns.pairplot(iris, hue="species", palette="husl", corner=True)
st.pyplot(fig1)

# Heatmap de corrélation
st.write("### Matrice de corrélation des caractéristiques")
fig2, ax2 = plt.subplots(figsize=(8, 6))
correlation_matrix = iris.iloc[:, :-1].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", ax=ax2, cbar=True)
ax2.set_title("Matrice de corrélation")
st.pyplot(fig2)

# Analyse de Clustering (K-means)
st.write("### Clustering non supervisé avec K-Means")

# Standardiser les données
scaler = StandardScaler()
iris_scaled = scaler.fit_transform(iris.iloc[:, :-1])

# Appliquer K-Means avec 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
iris["cluster"] = kmeans.fit_predict(iris_scaled)

# Visualisation des clusters
fig3, ax3 = plt.subplots(figsize=(8, 6))
sns.scatterplot(x=iris_scaled[:, 2], y=iris_scaled[:, 3], hue=iris["cluster"], palette="viridis", ax=ax3, s=100)
ax3.set_title("Clustering des données (K-Means)")
ax3.set_xlabel("Caractéristique 3 (standardisée)")
ax3.set_ylabel("Caractéristique 4 (standardisée)")
st.pyplot(fig3)

# Comparaison avec les espèces réelles
st.write("### Comparaison des clusters avec les espèces réelles")
comparison_table = pd.crosstab(iris["cluster"], iris["species"])
st.write(comparison_table)

# Signature
st.write("---")
st.write("Application développée par **Amaral**")
