from torch.utils.data import Dataset
import torch
import src.config as config

class PatientDataset(Dataset):
    def __init__(self, dataset, tokenizer, label_set):
        '''Init 
        data
        len
        label_map
        tokenizer
        '''
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.len = len(dataset)
        self.label_map = {label: i for i, label in enumerate(label_set)}
        self.id2label = list(label_set)
        self.max_len = config.MAX_LEN

    def __getitem__(self, idx):
        '''Return (tokens_tensor, segments_tensor, label_tensor)
        '''
        text = self.dataset[idx]['tokens']
        labels = [self.label_map[label] for label in self.dataset[idx]['labels']]

        # Because of the dataset, we use tokenizer seperately.
        # Normally you throw the whole text into tokenizer.
        word_pieces = []
        label_ids = []
        for char, l in zip(text, labels):
            token = self.tokenizer.tokenize(char)
            # Handle empty output by toknizer. (See sample 1002)
            if len(token) != 0:
                word_pieces.extend(token)
                label_ids.append(l)
        assert len(word_pieces) == len(label_ids)

        # Contrain max_len. -2 for 'CLS', 'SEP'.
        word_pieces = word_pieces[:self.max_len-2]
        label_ids = label_ids[:self.max_len-2]

        # Append special token 'CLS', 'SEP'.
        word_pieces = ['[CLS]'] + word_pieces + ['[SEP]']
        label_ids = [self.label_map['O']] + label_ids + [self.label_map['O']]

        # Convert token to id in the tokenizer dictionary
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)

        # Segment and mask.
        segments = [0] * len(ids)
        masks = [1] * len(ids)

        # Padding
        padding_len = self.max_len - len(ids)
        ids = ids + ([0] * padding_len)
        label_ids = label_ids + ([0] * padding_len)
        segments = segments + ([0] * padding_len)
        masks = masks + ([0] * padding_len)

        label_tensor = torch.tensor(label_ids, dtype=torch.long)
        tokens_tensor = torch.tensor(ids, dtype=torch.long)
        segments_tensor = torch.tensor(segments, dtype=torch.long)
        masks_tensor = torch.tensor(masks, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, masks_tensor, label_tensor)

    def __len__(self):
        return self.len
