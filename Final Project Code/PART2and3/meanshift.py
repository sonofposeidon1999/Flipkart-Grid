from sklearn.manifold import TSNE
from sklearn.cluster import MeanShift
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
scaler = StandardScaler()
def clustering(emb):
  temp = scaler.fit_transform(emb)
  Y = TSNE(n_components=2).fit_transform(temp)
  cluster_ms = MeanShift(bandwidth = 3,max_iter='200',cluster_all=False).fit(Y)
  y_ms = cluster_ms.predict(Y)

  plt.figure
  plt.scatter(Y[:,0], Y[:, 1], c=y_ms, s=50, cmap='viridis')
  #centers = kmeans.cluster_centers_
  #plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, alpha=1)
  plt.show()

  return y_ms