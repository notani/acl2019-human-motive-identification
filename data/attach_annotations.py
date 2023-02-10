from os import path

import pandas as pd


NUM_SPLITS = 3

SEMEVAL_FILES = {
    'restaurant': 'semeval2016/subtask1/en/restaurants_train_v2.tsv',
    'laptop': 'semeval2016/subtask1/en/laptops_train_v2.tsv',
}

for domain in ['restaurant', 'laptop']:
    for i in range(NUM_SPLITS):
        for split in ['train', 'valid', 'test', 'train+valid']:
            path_ann = path.join(domain, str(i), f'{split}.tsv')
            ann = pd.read_csv(path_ann, sep='\t')
            path_in = SEMEVAL_FILES[domain]
            df = pd.read_csv(path_in, sep='\t')
            out = pd.merge(df, ann, on='id', how='right')
            path_out = path.join(domain, str(i), f'{split}.raw.tsv')
            print(f'Write {len(out)} rows to {path_out}')
            out.to_csv(path_out, sep='\t', index=False)
