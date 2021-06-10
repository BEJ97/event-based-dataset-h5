from os.path import isfile


from load_dat_prophesee import * # ncars(.dat)
# from load_events import *  # n-mnist(.bin) cifar10-dvs(.aedat)
import h5py
import os
import os.path
from tqdm import tqdm

"""
N-MNIST:用aertb中函数完成：提取文件夹下样本，将样本输入至load_events函数中，输出一个样本对应的矩阵，将矩阵按照类别存储至h5文件中
N-CARS:load_events：用 https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox 的代码完成对样本的读取，输出对应矩阵
CIFAR10-DVS:
"""


def create_hdf5_dataset(dataset_name, file_or_dir, ext, ):
    """
        Creates an HDF5 file with the specified name, for a parent
        directory containing .dat files. It will create a different
        group for each subdirectory

        Params
        ------
        :param dataset_name: the name of the HDF5 file with file extension
        :param parent_dir: the path pointing to the parent directory
                            where the dat files reside
        :param polarities: indicates the polarity encoding for the
                            data, it can be [0,1] or [-1,1]

    """

    with h5py.File(dataset_name, 'w') as fp:

        # if we are dealing with only one file
        if isfile(file_or_dir):
            fname = os.path.split(file_or_dir)[1].split('.')[0]
            g = fp.create_group('root')
            events = load_dat_events(file_or_dir)
            g.create_dataset(f'{fname}', data=events, compression=8)

        # else we are dealing with directories
        else:
            _add_all_files(fp, file_or_dir, 'root', ext)

            # Navigate subdirectories
            sub_dirs = [f.name for f in os.scandir(file_or_dir) if f.is_dir()]
            if '.Ds_Store' in sub_dirs: sub_dirs.remove('.Ds_Store')

            # logging.info(f'Processing directories: {sub_dirs} ')
            # for each subdirectory add all_files
            for folder in sub_dirs:
                _add_all_files(fp, os.path.join(file_or_dir, folder), folder, ext)


def _add_all_files(fp, dir_path, dir_name, ext):
    """
        Supporting function for creating a dataset
    """

    # logging.info(f'Processing {dir_path}')

    # Get all file names
    all_files = [f for f in os.scandir(dir_path)]
    valid_files = [f.name for f in all_files if os.path.splitext(f)[1] == f'.{ext}']

    # logging.info(f'Files: {valid_files}')

    if len(valid_files) > 0:

        group = fp.create_group(dir_name)
        # logging.info(f'Found the following valid files {valid_files} in {dir_path}')

        for file in tqdm(valid_files, desc=f'Dir: {dir_name}', unit='file'):
            events = load_dat_events(os.path.join(dir_path, file))

            group.create_dataset(f"{file.split('.')[0]}", data=events, compression=8)

# create_hdf5_dataset(dataset_name, file_or_dir, ext)
# create_hdf5_dataset('cifar10-dvs.h5','D:/CIFAR10-DVS/CIFAR10-DVS','aedat')
