
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
scaler = StandardScaler()
def clustering(emb):
  temp = scaler.fit_transform(emb)
  Y = TSNE(n_components=2).fit_transform(temp)
  kmeans = KMeans(n_clusters=7,init = 'k-means++',n_init=20, max_iter=500,algorithm='elkan')
  kmeans.fit(Y)
  y_kmeans = kmeans.predict(Y)

  plt.figure
  plt.scatter(Y[:,0], Y[:, 1], c=y_kmeans, s=50, cmap='viridis')
  centers = kmeans.cluster_centers_
  plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1)
  plt.show()

  return y_kmeans
