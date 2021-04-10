from collections import OrderedDict
from typing import Callable, Optional
import torch
import torchvision
from torchvision.datasets import VisionDataset
from torchvision.transforms.transforms import ToTensor
import common
import pandas as pd
import dicom_processing
import numpy as np
import os
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plt

class UnbalancedDataset(VisionDataset):

    def __init__(self, df_path: str, 
                        root: str, 
                        transform=None, 
                        target_transform=None,
                        loader=dicom_processing.auto_loader) -> None:

        super().__init__(root, transforms=None, transform=transform, target_transform=target_transform)

        self.df_path = df_path
        self.df = pd.read_feather(df_path)
        self.loader = loader
        del self.df['index']


        self.classes = sorted(self.df['category'].unique())                 # make sure that training and validation set have same classes!!! in same order!!
        self.class_to_id = {x:i for i,x in enumerate(self.classes)}

    def __getitem__(self, index: int):
        
        item = self.df.iloc[index]

        X = self.loader(os.path.join(self.root, item['category'],item['filename']))
        if self.transform is not None: X = self.transform(X)
            
        y = torch.tensor(self.class_to_id[item['category']])
        
        if self.target_transform is not None: y = self.target_transform(y)

        return X, y

    def get_class_ratios(self):
        # returns the ratio of each class in the order of class_to_id
        x = self.df.groupby('category')['hash'].nunique()

        rs = torch.tensor([x[r] for r in self.class_to_id])

        return rs / rs.sum()

    def __len__(self) -> int:
        return len(self.df)

class UnbalancedSelfSupervisedRotationalDataset(UnbalancedDataset):
    # 0 is 0 deg, 1 is 90 deg, 2 is 180 deg, 3 is 270 deg 

    def __init__(self, df_path: str, 
                        root: str, 
                        transform=None, 
                        target_transform=None,
                        loader=dicom_processing.auto_loader) -> None:
        super().__init__(df_path, root, transform=transform, target_transform=target_transform, loader=loader)

    def __getitem__(self, index: int):
        
        item = self.df.iloc[index]

        X = self.loader(os.path.join(self.root, item['category'],item['filename']))
        if self.transform is not None: X = self.transform(X)

        rot = torch.tensor(random.randint(0,3))

        y = X.rot90(k=rot.item(), dims=(1,2))

        cat = torch.tensor(self.class_to_id[item['category']])
        
        if self.target_transform is not None: y = self.target_transform(y)

        return X, y, rot, cat

class UnetDataset(UnbalancedDataset):

    def __init__(self, df_path: str, 
                        root: str,
                        map_root: str,
                        transform=None, 
                        target_transform=None,
                        loader=dicom_processing.auto_loader,
                        target_loader=dicom_processing.segmentation_image_loader,
                        allow_non_segmentated_images=True,      # if true, will return ng maps and films without ng tubes with blank maps
                        blank_map_dims=(512,512)) -> None:      # before any transforms
        super().__init__(df_path, root, transform=transform, target_transform=target_transform, loader=loader)


        self.map_root = map_root
        self.target_loader = target_loader
        self.allow_non_segmentated_images = allow_non_segmentated_images
        self.blank_map_dims = blank_map_dims

        if not self.allow_non_segmentated_images:
            self.df = self.df[self.df['ng_roi_filename'].notnull()]
        else:
            #allow no ng tube and ng tubes with ng_roi_filename
            self.df = self.df[(self.df['ng_roi_filename'].notnull() | (self.df['category'] == 'NO_NG'))]

    def __getitem__(self, index: int):
        
        item = self.df.iloc[index]

        X = self.loader(os.path.join(self.root, item['category'],item['filename']))
        if self.transform is not None: X = self.transform(X)
        
        cat = torch.tensor(self.class_to_id[item['category']])

        if item['ng_roi_filename'] is None:
            # return blank target
            y = np.zeros(self.blank_map_dims)
        else:
            y = self.target_loader(os.path.join(self.map_root,item['ng_roi_filename']))

        if self.target_transform is not None: y = self.target_transform(y)

        return X, y, cat








if __name__ == '__main__':
    print('debugging')
    d = UnetDataset(common.TRAIN_DF_PATH, 
                        root=common.SATO_IMAGES_ROOT_PATH, 
                        map_root=common.NG_ROI_ROOT_PATH,
                        loader=dicom_processing.auto_loader,
                        transform=transforms.Compose([
                            ToTensor()
                        ]),
                        target_transform=transforms.Compose([
                            ToTensor()
                        ])
                        )

    for r in range(0, 100):
        x, y, cat = d.__getitem__(r)

        print(x.shape)
        print(y.shape)
        print(cat)
        print()

        plt.subplot(1,2,1)
        plt.imshow(x[0])
        plt.subplot(1,2,2)
        plt.imshow(y[0])

        plt.show()















