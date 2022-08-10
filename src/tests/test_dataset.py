from transformers import BertTokenizer
from torch.utils.data import DataLoader
import torch

from src.dataset import PatientDataset
from src.utils import load_dataset
import src.config as config

def test_patient_dataset():
    dataset, label_set = load_dataset('data/training.txt')
    tokenizer = BertTokenizer.from_pretrained(config.PRETRAINED_MODEL_NAME)
    train_size = int(len(dataset) * 0.01)
    trainset = PatientDataset(dataset[:train_size], tokenizer, label_set)

    sample_idx = 0
    tokens_tensor, segments_tensor, masks_tensor, label_tensor = trainset[sample_idx]
    assert tokens_tensor.shape == torch.Size([config.MAX_LEN])
    assert segments_tensor.shape == torch.Size([config.MAX_LEN])
    assert masks_tensor.shape == torch.Size([config.MAX_LEN])
    assert label_tensor.shape == torch.Size([config.MAX_LEN])

    trainloader = DataLoader(trainset, batch_size=config.BATCH_SIZE)
    data = next(iter(trainloader))
    tokens_tensors, segments_tensors, masks_tensors, label_tensors = data
    assert tokens_tensors.shape == torch.Size([config.BATCH_SIZE, config.MAX_LEN])
    assert segments_tensors.shape == torch.Size([config.BATCH_SIZE, config.MAX_LEN])
    assert masks_tensors.shape == torch.Size([config.BATCH_SIZE, config.MAX_LEN])
    assert label_tensors.shape == torch.Size([config.BATCH_SIZE, config.MAX_LEN])
