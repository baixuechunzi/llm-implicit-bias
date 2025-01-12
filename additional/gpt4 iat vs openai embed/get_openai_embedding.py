import pandas as pd
import json
from utils import *
# from openai import OpenAI
from openai import AzureOpenAI

client = AzureOpenAI(
    api_key='###',
    api_version="###",
    azure_endpoint='###'
)


def words_to_embeddings(words, word_type='target', model="text-embedding-3-small"):
    embeddings = {}
    for word in words:
        if word_type == 'target':
            to_embed = "Here is a list of words. For each word pick a word — " + word  # our prompt
        elif word_type == 'attribute':
            to_embed = " — and write it after the word. The words are " + word  # our prompt

        embeddings[word] = client.embeddings.create(input=[to_embed], model=model).data[0].embedding

    return embeddings


stimuli_df = pd.read_csv('iat_stimuli_updated.csv')
default_groups = stimuli_df['A'].dropna().str.lower().tolist()  # default
stigma_groups = stimuli_df['B'].dropna().str.lower().tolist()  # stigma

targets = set(default_groups + stigma_groups)
attributes = set(stimuli_df['C'].dropna().str.lower().tolist())

targets_dict = words_to_embeddings(targets, word_type='target', model="text-embedding-3-large")
attributes_dict = words_to_embeddings(attributes, word_type='attribute', model="text-embedding-3-large")

text_embed3_large = {**targets_dict, **attributes_dict}

with open('text_embed3_large.json', 'w') as f:
    json.dump(text_embed3_large, f, indent=4)
