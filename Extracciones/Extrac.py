import h5py
import numpy as np
import gc
from models.i3d.extract_i3d import ExtractI3D
from utils.utils import build_cfg_path
from omegaconf import OmegaConf
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.cuda.get_device_name(0)

# Select the feature type
feature_type = 'i3d'

# Load and patch the config
args = OmegaConf.load(build_cfg_path(feature_type))

with open('video_paths_val.txt', 'r') as file: 
    video_paths = [line.strip() for line in file.readlines()[800:856]]

args.video_paths = video_paths

#args.video_paths = ['../train_subset/00EMZGzlhek_000018_000028.mp4']
# args.show_pred = True
# args.stack_size = 24
# args.step_size = 24
# args.extraction_fps = 25
args.flow_type = 'raft'
# args.streams = 'flow'

extractor = ExtractI3D(args)

for video_path in args.video_paths:  
    print(f'Extrayendo características para {video_path}')
    
    try:
        feature_dict = extractor.extract(video_path)
        
        if 'flow' in feature_dict:
            video_name = video_path.split('/')[-1]  
            flow_data = feature_dict['flow']
            
            if isinstance(flow_data, np.ndarray) and flow_data.dtype.kind in {'i', 'f'}:  
                with h5py.File('extracted_flow_features_val.h5', 'a') as hdf_file:
                    hdf_file.create_dataset(f'{video_name}/flow', data=flow_data)
                print(f'Flow almacenado para {video_name}')
            else:
                print(f'Los datos de flujo para {video_name} no son numéricos.')
                
    except Exception as e:
        print(f'Error al procesar {video_path}: {e}')
    
    del feature_dict
    gc.collect()