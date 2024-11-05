import csv
import os
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

model_cache = 'model_cache/'

def generate_and_get_embeddings(prompt, tokenizer, model, targets, attributes):
    # Tokenize the prompt
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_words = tokenizer(prompt, return_tensors="pt").to('cuda:1')

    # Inference not training
    with torch.no_grad():
        outputs = model(**tokenized_words, output_hidden_states=True)
        hidden_states = outputs.hidden_states

        output_sequences = model.generate(**tokenized_words, max_new_tokens=600, temperature=0.3,
                                          eos_token_id=tokenizer.eos_token_id,
                                          pad_token_id=tokenizer.eos_token_id)

    response = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

    # Dictionary to store embeddings for each word across all layers
    embeddings = {}

    # Get token index for the targets in the tokenized input
    target_index0 = tokenized_words['input_ids'][0].tolist().index(tokenizer.encode(' ' + targets[0])[1])
    target_index1 = tokenized_words['input_ids'][0].tolist().index(tokenizer.encode(' ' + targets[1])[1])

    target_index_attributes = []
    for attribute in attributes:
        target_index_attributes.append(
            tokenized_words['input_ids'][0].tolist().index(tokenizer.encode(' ' + attribute)[1]))

    # Extract embeddings for the specified token from all layers
    target_embedding0 = [hidden_states[layer][0, target_index0, :] for layer in range(len(hidden_states))]
    target_embedding1 = [hidden_states[layer][0, target_index1, :] for layer in range(len(hidden_states))]

    target_embeddings = []
    for target_index_attribute in target_index_attributes:
        target_embeddings.append(
            [hidden_states[layer][0, target_index_attribute, :] for layer in range(len(hidden_states))])

    # Store in the dictionary
    embeddings[targets[0]] = target_embedding0
    embeddings[targets[1]] = target_embedding1
    for attribute, target_embedding in zip(attributes, target_embeddings):
        embeddings[attribute] = target_embedding

    return response, embeddings


# Load the model and tokenizer
model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map='cuda:1',
                                          cache_dir=model_cache)
model = AutoModelForCausalLM.from_pretrained(model_name, output_hidden_states=True, return_dict_in_generate=True,
                                             device_map='cuda:1', torch_dtype=torch.bfloat16,
                                             cache_dir=model_cache)

# read in tests
with open('./prompts.csv', mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.DictReader(file)

    exec_count_per_category = defaultdict(int)

    for i, line in enumerate(tqdm(csv_reader, desc="Processing CSV")):

        prompt = line['prompt']
        targets = [line['group0'], line['group1']]
        attributes = line['attributes'].split(', ')
        response, embeddings = generate_and_get_embeddings(prompt, tokenizer, model, targets, attributes)
        response_without_prompt = response[len(prompt):]

        category = line['category']
        exec_count_per_category[category] += 1

        prompt_path = f"raw_result/{category}/{exec_count_per_category[category]}/prompt"
        os.makedirs(os.path.dirname(prompt_path), exist_ok=True)
        with open(prompt_path, 'w') as f:
            f.write(prompt)

        response_path = f"raw_result/{category}/{exec_count_per_category[category]}/response"
        os.makedirs(os.path.dirname(response_path), exist_ok=True)
        with open(response_path, 'w') as f:
            f.write(response_without_prompt)

        for key in embeddings:
            # obtain the embedding for the attribute
            numpy_list = [tensor.to(dtype=torch.float16).cpu().numpy() for tensor in embeddings[key]]

            if key in targets:
                target_attribute_path = f"raw_result/{category}/{exec_count_per_category[category]}/target_{key}"
            elif key in attributes:
                target_attribute_path = f"raw_result/{category}/{exec_count_per_category[category]}/attribute_{key}"
            else:
                raise ValueError(f"{key} is neither in {targets} nor {attributes}")
            os.makedirs(os.path.dirname(target_attribute_path), exist_ok=True)
            np.savez(target_attribute_path, *numpy_list)

            '''
            load it use:
            
            import numpy as np

            # Load the .npz file
            loaded = np.load('tensors.npz')

            # Access arrays by index (e.g., 'arr_0', 'arr_1', etc.)
            for i in range(len(loaded.files)):
                print(loaded[f'arr_{i}'])
            '''