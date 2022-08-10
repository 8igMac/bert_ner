from tqdm import tqdm

def load_dataset(filepath: str):
    """
    Return a list of json with attributes:
    tokens: string
    labels: a list of string

    and label set
    """
    dataset = list()
    label_set = set()
    with open(filepath) as f:
        tokens = ''
        labels = list()
        
        for line in tqdm(f):
            splits = line.split()
            if len(splits) == 2:
                # Collect tokens and labels.
                char, label = splits
                if char != '*':
                    tokens += char
                    labels.append(label)
                    label_set.add(label)
            else:
                # Append tokens, labels to data.
                dataset.append({
                'tokens': tokens,
                'labels': labels
                })
                # Clear tokens and labels.
                tokens = ''
                labels = []

    return dataset, label_set
