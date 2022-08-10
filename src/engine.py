from tqdm import tqdm
import torch
from sklearn.metrics import f1_score

import src.acc_count as acc_count

def run_epoch(dataloader, model, device, optimizer = None, train = False):
    """ Run an epoch.
    Return
        - epoch accuracy.
        - epoch f1.
        - epoch loss per batch.
    """
    if train:
        model.train()
    else:
        model.eval()

    final_loss = 0
    preds = []
    targets = []
    for data in tqdm(dataloader, total=len(dataloader)):
        tokens_tensors, segments_tensors, masks_tensors, labels = [t.to(device) for t in data]
        if train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(train):
            outputs = model(
                input_ids=tokens_tensors, 
                token_type_ids=segments_tensors,
                attention_mask=masks_tensors,
                labels=labels,
            )
            loss = outputs.loss
            _, pred = torch.max(outputs.logits, 2) # batch, seq_len, labels
            if train:
                loss.backward()
                optimizer.step()

        # Remove padding when calculating f1 and acc.
        mask = (masks_tensors == 1)  # Convert to Boolean tensor.
        assert mask.shape == masks_tensors.shape
        pred = torch.masked_select(pred.flatten(), mask.flatten())
        labels = torch.masked_select(labels.flatten(), mask.flatten())

        final_loss += loss
        preds += pred.tolist()
        targets += labels.tolist()

    f1 = f1_score(targets, preds, average='weighted')

    # Label id to label.
    pred_labels = [dataloader.dataset.id2label[p] for p in preds]
    target_labels = [dataloader.dataset.id2label[t] for t in targets]
    tp, total = acc_count.acc(pred_labels, target_labels)
    acc = tp/total

    return acc, f1, final_loss / len(dataloader)
