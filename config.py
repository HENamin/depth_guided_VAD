import os

### Architecture
clips_length = 5
rgb_tags = True   # rgb_tags = False
img_channel = 3 if rgb_tags else 1
img_size = [256, 256]


### training parameters
frame_interval = 1
batch_size = 4
pretrain_batches = 4  # 5000
compact_start = pretrain_batches//4 # after quarter of training start to compact


seed = 107

AE_struct = [1,1,1]
temporal_struct = [1,1,1]
AE_layer_nums = 3
temporal_channel_in = 1
temporal_layer_nums = 3
AE_channel_out = img_channel


### eval setting
eval_batches = 4

compact = False  # Set to True if you want to enable compactness feature

testpath = 'Avenue\\Avenue_Dataset\\Test'
