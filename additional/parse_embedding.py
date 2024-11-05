import json
from pathlib import Path

from tqdm import tqdm

from utils import *

exec_counts = 10
layers = 33

pos_neg_dict = csv_to_dict('global_stimuli.csv')
pos_words = set([s.lower() for s in pos_neg_dict['POS']])
neg_words = set([s.lower() for s in pos_neg_dict['NEG']])

raw_result_path = Path("raw_result")
categories = [d.name for d in raw_result_path.iterdir() if d.is_dir()]

bias = {}
for category in tqdm(categories, desc="Processing CSV"):

    bias_per_category = {}
    for exec_count in range(1, exec_counts + 1):
        dir_path = Path(f"raw_result/{category}/{exec_count}/")
        # go through targets
        targets = [file.name[len("target_"):-4] for file in dir_path.iterdir() if
                   file.name.startswith("target_")]

        if targets[0].lower() in pos_words and targets[1].lower() in neg_words:
            pos_target = targets[0]
            neg_target = targets[1]
        elif targets[1].lower() in pos_words and targets[0].lower() in neg_words:
            pos_target = targets[1]
            neg_target = targets[0]
        else:
            raise ValueError(f'{targets[0]} or {targets[1]} not in {pos_words} or {neg_words}')

        # go through attributes
        attributes = [file.name[len("attribute_"):-4] for file in dir_path.iterdir() if
                      file.name.startswith("attribute_")]

        pos_attributes = []
        neg_attributes = []
        for attribute in attributes:
            if attribute in pos_words:
                pos_attributes.append(attribute)
            elif attribute in neg_words:
                neg_attributes.append(attribute)
            else:
                raise ValueError(f'{attribute} is neither in {pos_words} nor {neg_words}')

        emb_bias = {}
        for layer in range(layers):
            pos_target_path = f'raw_result/{category}/{exec_count}/target_{pos_target}.npz'
            pos_target_emb = get_embedding(pos_target_path, layer)

            neg_target_path = f'raw_result/{category}/{exec_count}/target_{neg_target}.npz'
            neg_target_emb = get_embedding(neg_target_path, layer)

            pos_attribute_embs = []
            for pos_attribute in pos_attributes:
                pos_attribute_path = f'raw_result/{category}/{exec_count}/attribute_{pos_attribute}.npz'
                pos_attribute_embs.append(get_embedding(pos_attribute_path, layer))

            neg_attribute_embs = []
            for neg_attribute in neg_attributes:
                neg_attribute_path = f'raw_result/{category}/{exec_count}/attribute_{neg_attribute}.npz'
                neg_attribute_embs.append(get_embedding(neg_attribute_path, layer))

            # computer weat
            emb_bias[layer] = weat_score(pos_target_emb[None, :],
                                         neg_target_emb[None, :],
                                         pos_attribute_embs,
                                         neg_attribute_embs)

        bias_per_category[exec_count] = emb_bias

    bias[category] = bias_per_category

with open('emb_bias.json', 'w') as f:
    json.dump(bias, f, indent=4)
