import common
import models
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from custom_dataset import LineSafeDataset
import dicom_processing
import torchvision.transforms as transforms
import custom_transforms 
from contextlib import nullcontext
import matplotlib.pyplot as plt
import torch.optim as optim
      

if __name__ == '__main__':

    common.ensure_reproducibility(common.ENSURE_REPRODUCIBILITY)  

    transform = transforms.Compose([
                                
                                transforms.ToTensor(),
                                custom_transforms.ToMultiChannel(3),
                                transforms.Resize((256,256)),
                            ])

    train_dataset = LineSafeDataset(common.TRAIN_DF_PATH, 
                            root=common.SATO_IMAGES_ROOT_PATH, 
                            ng_roi_root=common.NG_ROI_ROOT_PATH,
                            loader=dicom_processing.auto_loader,
                            target_loader=None,
                            transform=transform,
                            target_transform=None,
                            return_what='all')

    validate_dataset = LineSafeDataset(common.VALIDATE_DF_PATH, 
                            root=common.SATO_IMAGES_ROOT_PATH, 
                            ng_roi_root=common.NG_ROI_ROOT_PATH,
                            loader=dicom_processing.auto_loader,
                            target_loader=None,
                            transform=transform,
                            target_transform=None,
                            return_what='all')

    train_dataloader = DataLoader(train_dataset, common.TRAIN_BATCH_SIZE, shuffle=common.TRAIN_SHUFFLE, num_workers=common.NUMBER_OF_WORKERS)
    validate_dataloader = DataLoader(validate_dataset, common.VALIDATE_BATCH_SIZE, shuffle=common.VALIDATE_SHUFFLE, num_workers=common.NUMBER_OF_WORKERS)
    
    model = models.Resnet(out_features=3).to(common.DEVICE) # NG_OK, NG_NOT_OK, NO_NG
    optimizer = optim.Adadelta(model.parameters(), lr=common.LR)
    criterion = nn.NLLLoss(weight=train_dataset.get_class_ratios()).to(common.DEVICE) # incase of unbalanced datasets




    for epoch in range(1, common.TRAIN_EPOCHS + 1):

        print(f'Epoch {epoch} of {common.TRAIN_EPOCHS}:')

        for training, dataset, dataloader in [(True, train_dataset, train_dataloader), 
                                                (False, validate_dataset, validate_dataloader)]:

            if training:
                context_manager = nullcontext()
                model.train()
                print('\tTraining:')
            else:
                context_manager = torch.no_grad()
                model.eval()
                print('\tEvaluating:')

            with context_manager:
                
                total_loss = 0.

                for i, (X, y) in enumerate(dataloader):
                    
                    X = X.to(common.DEVICE)
                    y = y.to(common.DEVICE)

                    if training: optimizer.zero_grad()

                    output = model(X)

                    loss = criterion(output, y)

                    if training:
                        loss.backward()
                        optimizer.step()

                    print(f'\r\tBatch: {i+1} of {len(dataloader)}: loss: {loss.item():.4f}', end='')
                    
                    total_loss += loss.item()

            print()
            print(f'\t Total loss: {total_loss:.4f}')



