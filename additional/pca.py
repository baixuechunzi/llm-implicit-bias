import json
from pathlib import Path

from sklearn.decomposition import PCA
from tqdm import tqdm

from utils import *


def all_but_the_top(embeddings, k=1):
    """
    Removes the top k principal components from the embeddings.

    Args:
    - embeddings (np.ndarray): The matrix of word embeddings (shape: num_words x embedding_dim).
    - k (int): The number of top components to remove.

    Returns:
    - np.ndarray: The processed embeddings with top k components removed.
    """
    # Step 1: Compute the mean of the embeddings
    mean_embedding = np.mean(embeddings, axis=0)

    # Step 2: Subtract the mean from the embeddings
    centered_embeddings = embeddings - mean_embedding

    # Step 3: Perform PCA
    pca = PCA(n_components=k)
    pca.fit(centered_embeddings)

    # Step 4: Get the top k components
    top_components = pca.components_

    # Step 5: Project the centered embeddings onto these components and subtract them
    top_projections = np.dot(centered_embeddings, top_components.T)
    top_reconstruction = np.dot(top_projections, top_components)

    # Step 6: Subtract the reconstruction from the centered embeddings
    processed_embeddings = centered_embeddings - top_reconstruction

    # Optional: Add the mean back to retain the original space of embeddings
    processed_embeddings += mean_embedding

    return processed_embeddings


# Example usage:
# embeddings should be a 2D numpy array (num_words x embedding_dim)
layers = 33
exec_counts = 10
k = 2

pos_neg_dict = csv_to_dict('global_stimuli.csv')
pos_words = set([s.lower() for s in pos_neg_dict['POS']])
neg_words = set([s.lower() for s in pos_neg_dict['NEG']])

raw_result_path = Path("raw_result")
categories = [d.name for d in raw_result_path.iterdir() if d.is_dir()]

bias_nullified = {}
for category in tqdm(categories, desc="Processing CSV"):

    bias_per_category_nullified = {}
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

        emb_bias_nullified = {}
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

            all_embs = np.array([pos_target_emb, neg_target_emb] + pos_attribute_embs + neg_attribute_embs)
            all_embs_nullified = all_but_the_top(all_embs, k)

            pos_target_emb_nullified = all_embs_nullified[0]
            neg_target_emb_nullified = all_embs_nullified[1]
            pos_attribute_embs_nullified = all_embs_nullified[2:len(pos_attribute_embs) + 2]
            neg_attribute_embs_nullified = all_embs_nullified[len(pos_attribute_embs) + 2:]
            assert (len(pos_attribute_embs) == len(pos_attribute_embs_nullified))
            assert (len(neg_attribute_embs) == len(neg_attribute_embs_nullified))

            emb_bias_nullified[layer] = weat_score(pos_target_emb_nullified[None, :],
                                                   neg_target_emb_nullified[None, :],
                                                   np.array(pos_attribute_embs_nullified),
                                                   np.array(neg_attribute_embs_nullified))

        bias_per_category_nullified[exec_count] = emb_bias_nullified

    bias_nullified[category] = bias_per_category_nullified

with open('emb_bias_nullified.json', 'w') as f:
    json.dump(bias_nullified, f, indent=4)
