import h5py
import numpy as np
import umap

class SVD:
    def __init__(self, n_components):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.Vt = None

    def fit(self, X):
        X_centered = X - np.mean(X, axis=0)
        self.U, self.S, self.Vt = np.linalg.svd(X_centered, full_matrices=False)

    def transform(self, X):
        X_centered = X - np.mean(X, axis=0)
        return np.dot(X_centered, self.Vt.T[:, :self.n_components])

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)
    
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

svd = SVD(n_components=50)
X_svd = svd.fit_transform(all_flow_data)

reducer = umap.UMAP(n_components=2, n_neighbors=15, random_state=42)
X_umap = reducer.fit_transform(X_svd)

with h5py.File('./Data_reduced_svd_test_p.h5', 'w') as hdf_file:
    for i, video_name in enumerate(video_names):  
        hdf_file.create_dataset(video_name, data=X_umap[i])

print("Datos vacios:", cont)
print("Datos reducidos:", X_umap.shape)
print("Datos reducidos:", X_umap[:3])