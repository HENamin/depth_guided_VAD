import numpy as np
import torch
import sys
from types import SimpleNamespace

# Add the parent directory of sc_depth_pl_master to the Python path
sys.path.append('put the address here//sc_depth_pl_master')

# Import modules from sc_depth_pl_master
from sc_depth_pl_master.config import get_opts, get_training_size
from sc_depth_pl_master.SC_Depth import SC_Depth
from sc_depth_pl_master.SC_DepthV2 import SC_DepthV2
from sc_depth_pl_master.SC_DepthV3 import SC_DepthV3
import sc_depth_pl_master.datasets.custom_transforms as custom_transforms

from sc_depth_pl_master.visualization import *

@torch.no_grad()
def depth_map(input_tensor):
    # Set the arguments directly in the script
    config_path = './sc_depth_pl_master/configs/v3/ddad.txt'
    ckpt_path = './sc_depth_pl_master/ckpts/ddad_scv3/epoch=99-val_loss=0.1438.ckpt'
    
    # Get options and convert them to a SimpleNamespace object
    hparams = SimpleNamespace(**get_opts(config_path))
    hparams.ckpt_path = ckpt_path
    
    # Select the model based on version
    if hparams.model_version == 'v1':
        system = SC_Depth(hparams)
    elif hparams.model_version == 'v2':
        system = SC_DepthV2(hparams)
    elif hparams.model_version == 'v3':
        system = SC_DepthV3(hparams)

    system = SC_DepthV3.load_from_checkpoint(hparams.ckpt_path, strict=False)

    model = system.depth_net
    device = torch.device("cuda")
    model.to(device)
    model.eval()

    training_size = get_training_size(hparams.dataset_name)

    # Define transformations
    inference_transform = custom_transforms.Compose([
        custom_transforms.RescaleTo(training_size),
        custom_transforms.ArrayToTensor(),
        custom_transforms.Normalize()
    ])
    

    import torch.nn.functional as F
    input_tensor_resized = F.interpolate(input_tensor, size=(256, 320), mode='bilinear', align_corners=False)
    input_tensor_normal = (input_tensor_resized - input_tensor_resized.mean() ) / input_tensor_resized.std()
    pred_depth = model(input_tensor_normal.to(device))

    # Depth data
    pred_depth = F.interpolate(pred_depth, size=(256, 256), mode='bilinear')

    return pred_depth #, vis_img

if __name__ == '__main__':
    # Example tensor input
    example_tensor = torch.randn([1, 3, 256, 256])
    
    # Run inference
    result = depth_map(example_tensor)
    
    # Access depth and visualization
    depth_map_result = result['depth']
    vis_image = result['vis']
