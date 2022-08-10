import time
import torch
from transformers import BertForTokenClassification, BertTokenizer
from torch.utils.data import DataLoader

from src.engine import run_epoch
from src.utils import load_dataset
from src.dataset import PatientDataset
import src.config as config

if __name__ == '__main__':
    # Load data.
    dataset, label_set = load_dataset('data/training.txt')
    tokenizer = BertTokenizer.from_pretrained(config.MODEL_PATH)
    test_size = int(len(dataset) * config.TEST_RATIO)
    testset = PatientDataset(dataset[-test_size:], tokenizer, label_set)
    testloader = DataLoader(testset, batch_size=config.BATCH_SIZE)

    # Load model.
    model = BertForTokenClassification.from_pretrained(config.MODEL_PATH)

    # Check device.
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f'device: {device}')

    # Evaluate.
    start = time.time()
    acc, f1, loss = run_epoch(testloader, model, device)
    end = time.time()
    print(f'F1 score: {f1}')
    print(f'NER acc: {acc}')
    print(f'Number of sentences: {len(testset)}')
    print(f'Total time(s): {end - start}')
    print(f'Time(s) per sentences: {(end - start) / len(testset)}')
