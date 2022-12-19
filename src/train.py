from transformers import BertForTokenClassification, BertTokenizer
from torch.utils.data import DataLoader
import torch
from tqdm import tqdm
import numpy as np
import time
from torch.utils.tensorboard import SummaryWriter

from src.utils import load_dataset
from src.dataset import PatientDataset
import src.config as config
from src.engine import run_epoch

if __name__ == '__main__':
    # Load data.
    dataset, label_set = load_dataset('data/training.txt')

    # Prepare dataset and dataloader.
    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    train_size = int(len(dataset) * config.TRAIN_RATIO)
    trainset = PatientDataset(dataset[:train_size], tokenizer, label_set)
    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE)
    vali_size = int(len(dataset) * config.VALI_RATIO)
    valiset = PatientDataset(dataset[train_size:train_size + vali_size], tokenizer, label_set)
    valiloader = DataLoader(valiset, batch_size=config.BATCH_SIZE)
    print(f'training data: {len(trainset)}')

    # Create model.
    model = BertForTokenClassification.from_pretrained(
        config.PRETRAINED_MODEL_NAME,
        num_labels=len(label_set),
    )

    # Check device.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'device: {device}')

    # Training.
    writer = SummaryWriter()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    best_loss = np.inf
    start = time.time()
    for epoch in range(config.EPOCHS):
        train_acc, train_f1, train_loss = run_epoch(trainloader, model, device, optimizer, train=True) 
        vali_acc, vali_f1, vali_loss = run_epoch(valiloader, model, device) 
        print(f'[epoch {epoch+1}/{config.EPOCHS}, train loss: {train_loss}, vali loss: {vali_loss}, vali f1: {vali_f1}, vali acc: {vali_acc}]')

        # Write to tensorboard.
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/vali', vali_loss, epoch)
        writer.add_scalar('F1/train', train_f1, epoch)
        writer.add_scalar('F1/vali', vali_f1, epoch)
        writer.add_scalar('Acc/test', train_acc, epoch)
        writer.add_scalar('Acc/vali', vali_acc, epoch)

        if vali_loss < best_loss:
            model.save_pretrained(config.MODEL_PATH)
            tokenizer.save_vocabulary(config.MODEL_PATH)
            best_loss = vali_loss
    end = time.time()
    print(f'Training time(s): {end - start}')
