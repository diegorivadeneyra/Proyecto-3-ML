import h5py
import numpy as np


def PCA(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    covariance_matrix = np.cov(X_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, sorted_indices[:n_components]]
    X_reduced = np.dot(X_centered, components)
    return X_reduced



with h5py.File('./video_features/combined_flow_features.h5', 'r') as hdf_file:
    all_flow_data = []
    cont=0
    for video_name in hdf_file.keys():  
        flow = hdf_file[video_name]['flow'][()] 
        if flow.size == 0:
            print(f"Flujo vacío para {video_name}, se ignorará.")
            cont=cont+1
            continue

        flow_flattened = np.mean(flow, axis=0)
        all_flow_data.append(flow_flattened)    

all_flow_data = np.vstack(all_flow_data)

# Aplica PCA
X_pca = PCA(all_flow_data, n_components=2)  

with h5py.File('./Data_reduced_pca.h5', 'w') as hdf_file:
    
    hdf_file.create_dataset('reduced_flow_data', data=X_pca)

print("Datos vacios:", cont)
print("Datos reducidos:", X_pca.shape)
print("Datos reducidos:", X_pca[:5])