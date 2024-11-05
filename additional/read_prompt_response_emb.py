import csv
import json
from collections import defaultdict


# Writing dictionary to CSV
def dict_to_csv(dictionary, file_name):
    headers = dictionary.keys()
    rows = zip(*dictionary.values())

    with open(file_name, 'w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(headers)  # Write headers
        writer.writerows(rows)  # Write rows


with open('emb_bias.json', 'r') as f:
    embs = json.load(f)

table = defaultdict(list)
for category in embs.keys():
    for trial in embs[category].keys():
        table['category'].append(category)
        table['trial'].append(trial)

        with open(f'raw_result/{category}/{trial}/prompt', 'r') as f:
            table['prompt'].append(f.read())

        with open(f'raw_result/{category}/{trial}/response', 'r') as f:
            response = f.read()
            table['response'].append(response)

            # get task indices:
            try:
                task1_start_idx = response.index('Task 1:')
                task2_start_idx = response.index('Task 2:')
                task3_start_idx = response.index('Task 3:')
            except ValueError:
                table['response_task1'].append('bad response')
                table['response_task2'].append('bad response')
                table['response_task3'].append('bad response')

            table['response_task1'].append(response[task1_start_idx:task2_start_idx].strip())
            table['response_task2'].append(response[task2_start_idx:task3_start_idx].strip())
            table['response_task3'].append(response[task3_start_idx:].strip())

        for layer in embs[category][trial].keys():
            table[f'weat{layer}'].append(embs[category][trial][layer])

dict_to_csv(table, 'prompt_response_emb.csv')
