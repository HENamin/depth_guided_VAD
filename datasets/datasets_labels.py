import numpy as np
import os
from scipy.io import loadmat
# import scipy.io.loadmat as loadmat


def ped2_label(root_path = 'DATASET/'):
    import scipy.io as scio
    mat_file_path = os.path.join(root_path, 'UCSDPed2\\ped2.mat')
    video_file_path = os.path.join(root_path, 'UCSDPed2\\Test')
    testing_frames_list = os.listdir( video_file_path )
    testing_frames_list.sort()


    label_raw = scio.loadmat(mat_file_path, squeeze_me=True)['gt']
    if not len(testing_frames_list) == label_raw.shape[0]:
        print('error')
        return
    video_num = len(testing_frames_list)
    video_gt = []
    for idx in range(video_num):
        tmp_video_path = os.path.join(video_file_path, testing_frames_list[idx] )
        video_length = len( os.listdir( tmp_video_path ) )
        sub_video_gt = np.zeros((video_length,), dtype=np.int8)
        one_normal = label_raw[idx].item().__getitem__(0) -1
        sub_video_gt[one_normal] = 1
        video_gt.append(sub_video_gt)    
    return video_gt

def avenue_label(root_path = 'DATASET/'):
    import scipy.io as scio
    mat_file_path = os.path.join(root_path, 'avenue.mat')
    video_file_path = os.path.join(root_path, 'Avenue\\Avenue_Dataset\\Test')
    testing_frames_list = os.listdir( video_file_path )
    testing_frames_list.sort()
    label_raw = scio.loadmat(mat_file_path, squeeze_me=True)['gt']
    if not len(testing_frames_list) == label_raw.shape[0]:
        print('error')
        return
    video_num = len(testing_frames_list)
    video_gt = []
    for idx in range(video_num):
        tmp_video_path = os.path.join(video_file_path, testing_frames_list[idx] )
        video_length = len( os.listdir( tmp_video_path ) )
        sub_video_gt = np.zeros((video_length,), dtype=np.int8)
        abnormal_np = label_raw[idx].item().__getitem__(0) -1
        sub_video_gt[abnormal_np] = 1
        video_gt.append(sub_video_gt)    
    return video_gt

def gather_datasets_labels(root_path='/mnt/data/DataSet/datasets/', dataset='avenue'):
    datasets_labels = {}
    datasets_labels['avenue'] = avenue_label
    datasets_labels['ped2'] = ped2_label

    return datasets_labels[dataset](root_path)

if __name__ == '__main__':
    datasets_labels = gather_datasets_labels()