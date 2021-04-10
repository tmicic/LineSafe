from collections import namedtuple
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
import os
      

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
    
    MODEL_PATH = r'basic_resnet.model'

    model = models.Resnet(out_features=3).to(common.DEVICE) # NG_OK, NG_NOT_OK, NO_NG
    optimizer = optim.Adam(model.parameters(), lr=common.LR)
    criterion = nn.NLLLoss(weight=train_dataset.get_class_ratios()).to(common.DEVICE) # incase of unbalanced datasets

    # load model
    if MODEL_PATH != '':
        if os.path.exists(MODEL_PATH):
            print(f'Loading model ({MODEL_PATH})...')
            model = common.load_model_state(model, MODEL_PATH)
        else:
            print(f'Model file does not exist.')


    for epoch in range(0 if common.ALWAYS_VALIDATE_MODEL_FIRST else 1, common.TRAIN_EPOCHS + 1):    # do an epoch 0 if pre-val required

        if epoch > 0:
            print(f'Epoch {epoch} of {common.TRAIN_EPOCHS}:')
        else:
            print(f'Pre-evaluating model...')

        for training, dataset, dataloader in [(True, train_dataset, train_dataloader), 
                                                (False, validate_dataset, validate_dataloader)]:

            if epoch == 0 and training:
                # pre-eval therefore skip training
                continue

            if training:
                context_manager = nullcontext()
                model.train()
                print('\tTraining:')
            else:
                context_manager = torch.no_grad()
                model.eval()
                print('\tEvaluating:')

            stats = {'count':0, 'positive_count':0, 'negative_count':0, 'true_positives':0, 'true_negatives':0, 'false_positives':0, 'false_negatives':0 }

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

                    # sort out stats
                    stats['count'] += len(X)
                    stats['positive_count'] += (y==common.POSITIVE_CLASS).sum()
                    stats['negative_count'] += (y!=common.POSITIVE_CLASS).sum()
                    stats['true_positives'] += ((output.argmax(dim=1)==common.POSITIVE_CLASS) & (y==common.POSITIVE_CLASS)).sum()
                    stats['true_negatives'] += ((output.argmax(dim=1)!=common.POSITIVE_CLASS) & (y!=common.POSITIVE_CLASS)).sum()
                    stats['false_positives'] += ((output.argmax(dim=1)==common.POSITIVE_CLASS) & (y!=common.POSITIVE_CLASS)).sum()
                    stats['false_negatives'] += ((output.argmax(dim=1)!=common.POSITIVE_CLASS) & (y==common.POSITIVE_CLASS)).sum()

                    # calculations from https://en.wikipedia.org/wiki/Sensitivity_and_specificity - all as tensors
                    false_positive_rate = stats['false_positives'] / stats['negative_count']
                    false_negative_rate = stats['false_negatives'] / stats['positive_count']
                    sensitivity = 1 - false_negative_rate
                    specificity = 1 - false_positive_rate
                    accuracy = (stats['true_positives'] + stats['true_negatives']) / stats['count']
                    balananced_accuracy = (sensitivity + specificity) / 2 

                    print(f'\r\tBatch: {i+1} of {len(dataloader)}: loss: {loss.item():.4f}', end='')
                    
                    total_loss += loss.item()

            print()
            print(f'\t Total loss: {total_loss:.4f}\t Sens: {sensitivity.item()*100:.2f}%\t Spec: {specificity.item()*100:.2f}%\t Acc: {accuracy.item()*100:.2f}%\t Bal Acc: {balananced_accuracy.item()*100:.2f}%')



