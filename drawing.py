import numpy as np
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.pyplot as plt

# create default 'linesafe' colourmap for matplotlib
color_array = plt.get_cmap('viridis')(range(256))
color_array[:,-1] = np.linspace(0.0,1.0,256)
map_object = LinearSegmentedColormap.from_list(name='linesafe',colors=color_array)
plt.register_cmap(cmap=map_object)

def update_figure_unet(X, y, prediction):
    plt.clf()

    ax1 = plt.subplot(1,3,1)
    ax1.imshow(X.view(-1,256), cmap='gray')
    
    ax2 = plt.subplot(1,3,2)
    ax2.imshow(X.view(-1,256), cmap='gray')
    ax2.imshow(y.view(-1,256), cmap='linesafe', alpha=0.6)

    ax3 = plt.subplot(1,3,3)
    ax3.imshow(X.view(-1,256), cmap='gray')
    ax3.imshow(prediction.view(-1,256), cmap='linesafe', alpha=0.6)

    plt.pause(0.05)

def prevent_figure_close():
    plt.show()
    
if __name__ == '__main__':
    
    import common
    import custom_dataset, torch
    import torch.nn
    import torchvision.transforms as transforms
    from custom_dataset import UnetDataset
    import dicom_processing
    from torch.utils.data import DataLoader
    


    transform = transforms.Compose([
                                
                                transforms.ToTensor(),
                                transforms.Resize((256,256)),
                            ])

    target_transform = transforms.Compose([  
                                transforms.ToTensor(),
                                transforms.Resize((256,256)),
                            ])


    train_dataset = UnetDataset(common.TRAIN_DF_PATH, 
                            root=common.SATO_IMAGES_ROOT_PATH, 
                            map_root=common.NG_ROI_ROOT_PATH,
                            loader=dicom_processing.auto_loader,
                            transform=transform,
                            target_transform=target_transform,)

    train_dataloader = DataLoader(train_dataset, common.TRAIN_BATCH_SIZE, shuffle=common.TRAIN_SHUFFLE, num_workers=common.NUMBER_OF_WORKERS)


    X, y, cat = next(iter(train_dataloader))







    for a in range(len(X)):
        update_figure_unet(X[a], y[a], y[a])

    prevent_figure_close()

    




    
    
    