import h5py
import bisect
from pathlib import Path
from typing import List, Optional
from torch.utils.data import Dataset
import numpy as np
from numpy.typing import NDArray


list_path = List[Path]

class SingleShockDataset(Dataset):
    """Read single hdf5 file regardless of label, subject, and paradigm."""
    def __init__(self, file_path: Path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Extract datasets from file_path.

        param Path file_path: the path of target data
        param int window_size: the length of a single sample
        param int stride_size: the interval between two adjacent samples
        param float start_percentage: Index of percentage of the first sample of the dataset in the data file (inclusive)
        param float end_percentage: Index of percentage of end of dataset sample in data file (not included)
        '''
        self.__file_path = file_path
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__file: Optional[h5py.File] = None
        self.__length: int = 0
        self.__feature_size: Optional[List[int]] = None

        self.__subjects = []
        self.__global_idxes = []
        self.__local_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__file = h5py.File(str(self.__file_path), 'r')
        self.__subjects = [i for i in self.__file]

        global_idx = 0
        for subject in self.__subjects:
            self.__global_idxes.append(global_idx) # the start index of the subject's sample in the dataset
            subject_grp = self.__file[subject]
            assert isinstance(subject_grp, h5py.Group)
            eeg_dataset = subject_grp['eeg']
            assert isinstance(eeg_dataset, h5py.Dataset)
            subject_len: int = eeg_dataset.shape[1]
            # total number of samples
            total_sample_num = (subject_len-self.__window_size) // self.__stride_size + 1
            # cut out part of samples
            start_idx = int(total_sample_num * self.__start_percentage) * self.__stride_size 
            end_idx = int(total_sample_num * self.__end_percentage - 1) * self.__stride_size

            self.__local_idxes.append(start_idx)
            global_idx += (end_idx - start_idx) // self.__stride_size + 1
        self.__length = global_idx

        first_subject = self.__file[self.__subjects[0]]
        assert isinstance(first_subject, h5py.Group)
        first_eeg = first_subject['eeg']
        assert isinstance(first_eeg, h5py.Dataset)
        self.__feature_size = [i for i in first_eeg.shape]
        self.__feature_size[1] = self.__window_size

    @property
    def feature_size(self) -> Optional[List[int]]:
        return self.__feature_size

    def __len__(self) -> int:
        return self.__length

    def __getitem__(self, idx: int) -> NDArray[np.floating]:
        subject_idx = bisect.bisect(self.__global_idxes, idx) - 1
        item_start_idx = (idx - self.__global_idxes[subject_idx]) * self.__stride_size + self.__local_idxes[subject_idx]
        assert self.__file is not None
        subject_grp = self.__file[self.__subjects[subject_idx]]
        assert isinstance(subject_grp, h5py.Group)
        eeg_data = subject_grp['eeg']
        assert isinstance(eeg_data, h5py.Dataset)
        return eeg_data[:, item_start_idx:item_start_idx+self.__window_size]
    
    def free(self) -> None: 
        if self.__file:
            self.__file.close()
            self.__file = None
    
    def get_ch_names(self):
        assert self.__file is not None
        subject_grp = self.__file[self.__subjects[0]]
        assert isinstance(subject_grp, h5py.Group)
        eeg_data = subject_grp['eeg']
        assert isinstance(eeg_data, h5py.Dataset)
        return eeg_data.attrs['chOrder']


class ShockDataset(Dataset):
    """integrate multiple hdf5 files"""
    def __init__(self, file_paths: list_path, window_size: int=200, stride_size: int=1, start_percentage: float=0, end_percentage: float=1):
        '''
        Arguments will be passed to SingleShockDataset. Refer to SingleShockDataset.
        '''
        self.__file_paths = file_paths
        self.__window_size = window_size
        self.__stride_size = stride_size
        self.__start_percentage = start_percentage
        self.__end_percentage = end_percentage

        self.__datasets: List[SingleShockDataset] = []
        self.__length: int = 0
        self.__feature_size: Optional[List[int]] = None

        self.__dataset_idxes = []
        
        self.__init_dataset()

    def __init_dataset(self) -> None:
        self.__datasets = [SingleShockDataset(file_path, self.__window_size, self.__stride_size, self.__start_percentage, self.__end_percentage) for file_path in self.__file_paths]
        
        # calculate the number of samples for each subdataset to form the integral indexes
        dataset_idx = 0
        for dataset in self.__datasets:
            self.__dataset_idxes.append(dataset_idx)
            dataset_idx += len(dataset)
        self.__length = dataset_idx

        self.__feature_size = self.__datasets[0].feature_size

    @property
    def feature_size(self) -> Optional[List[int]]:
        return self.__feature_size

    def __len__(self) -> int:
        return self.__length

    def __getitem__(self, idx: int):
        dataset_idx = bisect.bisect(self.__dataset_idxes, idx) - 1
        item_idx = (idx - self.__dataset_idxes[dataset_idx])
        return self.__datasets[dataset_idx][item_idx]
    
    def free(self) -> None:
        for dataset in self.__datasets:
            dataset.free()
    
    def get_ch_names(self):
        return self.__datasets[0].get_ch_names()
