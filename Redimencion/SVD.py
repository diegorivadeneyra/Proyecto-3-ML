import h5py
import numpy as np

def SVD(X, n_components):
    X_centered = X - np.mean(X, axis=0)
    U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
    X_reduced = np.dot(X_centered, Vt[:n_components].T)
    return X_reduced

with h5py.File('./video_features/combined_flow_features.h5', 'r') as hdf_file:
    all_flow_data = []
    max_length = 4096  # Ajusta esto al tamaño máximo esperado de tus extracciones

    for video_name in hdf_file.keys():  
        flow = hdf_file[video_name]['flow'][()]
        if flow.size == 0:
            continue
        flow_flat = flow.flatten()
        
        # Ajustar la longitud rellenando con ceros o truncando
        if flow_flat.size < max_length:
            flow_flat = np.pad(flow_flat, (0, max_length - flow_flat.size))
        else:
            flow_flat = flow_flat[:max_length]
        
        all_flow_data.append(flow_flat)

all_flow_data = np.vstack(all_flow_data)

# Aplicar SVD
X_svd = SVD(all_flow_data, n_components=2)

with h5py.File('./Data_reduced_svd.h5', 'w') as hdf_file:
    
    hdf_file.create_dataset('reduced_flow_data', data=X_svd)

print("Datos reducidos:", X_svd.shape[0])
print("Datos reducidos:", X_svd[:5])