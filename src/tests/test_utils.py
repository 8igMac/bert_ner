from ..utils import load_dataset

def test_load_dataset():
    dataset, label_set = load_dataset('data/training.txt')
    first_data = {
        'tokens': "今天下午開始雙腳抽筋，現求治。",
        'labels': [
            'B-Tim', 'I-Tim', 'I-Tim', 'E-Tim', 'O', 'O', 
            'B-Org', 'E-Org', 'B-Sym', 'E-Sym', 'O', 'O', 
            'O', 'O', 'O',
        ],
    }
    label = {
        'B-Dep', 'I-Dep', 'E-Dep', 'S-Sym', 'I-Sym', 'O', 'E-Sym', 
        'B-Tim', 'B-Tre', 'E-Abb', 'I-Tre', 'S-Dis', 'I-Org', 
        'S-Tim', 'E-Tim', 'S-Exa', 'E-Dis', 'I-Dis', 'I-Abb', 
        'B-Sym', 'B-Med', 'B-Dis', 'S-Hea', 'E-Tre', 'I-Tim', 
        'B-Hea', 'I-Hea', 'E-Exa', 'E-Org', 'E-Med', 'B-Abb', 
        'I-Exa', 'B-Exa', 'E-Hea', 'S-Abb', 'I-Med', 'S-Org', 'B-Org',
    }
    assert len(dataset) == 65286
    assert dataset[0] == first_data
    assert len(label_set) == 38
    assert label_set == label
