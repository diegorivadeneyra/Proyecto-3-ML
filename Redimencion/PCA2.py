import h5py
import numpy as np
import umap

class PCA:

  def __init__(self, n_components):
    self.n_components = n_components
    self.components = None
    self.mean = None

  def fit(self, X):
    self.mean = np.mean(X, axis=0)
    X_ = X -  self.mean
    cov = np.cov(X_,rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)

    sort_idxs = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sort_idxs]
    eigenvectors = eigenvectors[:,sort_idxs]
    self.components = eigenvectors[:,:self.n_components]

  def transform(self, X):
    X_ = X - self.mean
    return np.dot(X_, self.components)

with h5py.File('./video_features/extracted_flow_features_test.h5', 'r') as hdf_file:
    all_flow_data = []
    video_names = []
    cont=0
    max_length = 1024
    for video_name in hdf_file.keys():  
        flow = hdf_file[video_name]['flow'][()] 
        if flow.size == 0:
            print(f"Flujo vacío para {video_name}, se ignorará.")
            cont=cont+1
            continue
        flow_flattened = np.mean(flow, axis=0).reshape(1, -1)
        all_flow_data.append(flow_flattened)    
        video_names.append(video_name)
all_flow_data = np.vstack(all_flow_data)

pca = PCA(50)
pca.fit(all_flow_data)
X_pca = pca.transform(all_flow_data)

reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
X_umap = reducer.fit_transform(X_pca)

with h5py.File('./Data_reduced_pca_test_p.h5', 'w') as hdf_file:
  for i, video_name in enumerate(video_names):  
    hdf_file.create_dataset(video_name, data=X_umap[i])

print("Datos vacios:", cont)
print("Datos reducidos:", X_umap.shape)
print("Datos reducidos:", X_umap[:3])