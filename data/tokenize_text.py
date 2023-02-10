#!/usr/bin/env python
# -*- coding: utf-8 -*-

from os import path

import pandas as pd
import spacy


NUM_SPLITS = 3
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

for domain in ['restaurant', 'laptop']:
    for i in range(NUM_SPLITS):
        for split in ['train', 'valid', 'test', 'train+valid']:
            filepath = path.join(domain, str(i), f'{split}.raw.tsv')
            df = pd.read_csv(filepath, sep='\t')
            df['text_tkn'] = df['text'].apply(lambda txt: ' '.join([t.text for t in nlp(txt)]))
            out_filepath = path.join(domain, str(i), f'{split}.tkn.tsv')
            print(f'Write {len(df)} rows to {out_filepath}')
            df.to_csv(out_filepath, sep='\t', index=False)
