import json
from pathlib import Path

import ast

from utils import *
import pandas as pd

pos_neg_dict = csv_to_dict('global_stimuli.csv')
pos_words = set([s.lower() for s in pos_neg_dict['POS']])
neg_words = set([s.lower() for s in pos_neg_dict['NEG']])

iat_df = pd.read_csv('temperature0_result_iat.csv')

emb = {}
with open('text_embed3_small.json', 'r') as f:
    emb['small'] = json.load(f)
with open('text_embed3_large.json', 'r') as f:
    emb['large'] = json.load(f)

for i, row in iat_df.iterrows():
    group0 = row['group0']
    group1 = row['group1']
    attributes = ast.literal_eval(row['attributes'])

    if group0.lower() in pos_words and group1.lower() in neg_words:
        pos_target = group0.lower()
        neg_target = group1.lower()
    elif group1.lower() in pos_words and group0.lower() in neg_words:
        pos_target = group1.lower()
        neg_target = group0.lower()
    else:
        raise ValueError(f'{group0} or {group1} not in {pos_words} or {neg_words}')

    # go through attributes
    pos_attributes = []
    neg_attributes = []
    for attribute in attributes:
        if attribute.lower() in pos_words:
            pos_attributes.append(attribute.lower())
        elif attribute.lower() in neg_words:
            neg_attributes.append(attribute.lower())
        else:
            raise ValueError(f'{attribute} is neither in {pos_words} nor {neg_words}')

    for trial in ['small', 'large']:
        pos_target_emb = np.array(emb[trial][pos_target])[None, :]
        neg_target_emb = np.array(emb[trial][neg_target])[None, :]

        pos_attribute_embs = []
        for pos_attribute in pos_attributes:
            pos_attribute_embs.append(emb[trial][pos_attribute])

        neg_attribute_embs = []
        for neg_attribute in neg_attributes:
            neg_attribute_embs.append(emb[trial][neg_attribute])

        # computer weat
        iat_df.at[i, f'weat_{trial}'] = weat_score(pos_target_emb,
                                                   neg_target_emb,
                                                   pos_attribute_embs,
                                                   neg_attribute_embs)

iat_df.to_csv('temperature0_result_iat_updated.csv')
