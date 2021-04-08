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

class LineSafeDataset(VisionDataset):

    def __init__(self, df_path: str, 
                        root: str, 
                        ng_roi_root: str=None, 
                        hilar_roi_root: str=None, 
                        return_what='all', 
                        transform=None, 
                        target_transform=None,
                        loader=dicom_processing.auto_loader,
                        target_loader=None) -> None:
        '''

        return_what: all = will return all images,
                     ng = will return images only if they have an ng map
                     hilar = will return images only if they have a hilar map
                     both = will return images only if they have hilar and ng map
                     ng_or_no_ng = will return images if they have an ng map or is a film that doesnt have an ng
        '''

        super().__init__(root, transforms=None, transform=transform, target_transform=target_transform)

        self.return_what = return_what
        self.df_path = df_path
        self.ng_roi_root = ng_roi_root
        self.hilar_roi_root = hilar_roi_root
        self.df = pd.read_feather(df_path)
        self.loader = loader
        self.target_loader = target_loader
        del self.df['index']


        self.classes = self.df['category'].unique()
        self.class_to_id = {x:i for i,x in enumerate(self.classes)}

        if self.return_what == 'all':
            pass
        elif self.return_what == 'ng':
            self.df = self.df[self.df['ng_roi_filename'].notnull()]
        elif self.return_what == 'hilar':
            self.df = self.df[self.df['hilar_roi_filename'].notnull()]
        else:
            raise NotImplementedError()

    def __getitem__(self, index: int):
        
        item = self.df.iloc[index]
        X = self.loader(os.path.join(self.root, item['category'],item['filename']))
        if self.transform is not None: X = self.transform(X)

        if self.return_what=='all':
            
            y = torch.tensor(self.class_to_id[item['category']])
            
            
            if self.target_transform is not None: y = self.target_transform(y)

            return X, y

        elif self.return_what == 'ng':

            y = self.target_loader(os.path.join(self.ng_roi_root,item['ng_roi_filename']))


            if self.target_transform is not None: y = self.target_transform(y)

            cat = torch.tensor(self.class_to_id[item['category']])

            return X, y, cat

        else:
            raise NotImplementedError()   

    def get_class_ratios(self):
        # returns the ratio of each class in the order of class_to_id
        x = self.df.groupby('category')['hash'].nunique()

        rs = torch.tensor([x[r] for r in self.class_to_id])

        return rs / rs.sum()


        

    def __len__(self) -> int:
        return len(self.df)

if __name__ == '__main__':
    print('debugging')

    d = LineSafeDataset(common.TRAIN_DF_PATH, 
                        root=common.SATO_IMAGES_ROOT_PATH, 
                        ng_roi_root=common.NG_ROI_ROOT_PATH,
                        loader=dicom_processing.auto_loader,
                        target_loader=dicom_processing.segmentation_image_loader,
                        transform=transforms.Compose([
                            ToTensor()
                        ]),
                        target_transform=transforms.Compose([
                            ToTensor()
                        ]),
                        return_what='all')

    print(d.class_to_id)
    print(d.get_class_ratios())
    exit()











