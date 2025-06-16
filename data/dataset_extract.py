from datasets import load_dataset, Dataset
import os
from tqdm import tqdm

# Define the target difficulties
# Difficulty levels: EASY (1), MEDIUM (2), HARD (3), HARDER (4), HARDEST (5)
target_difficulties = [0, 1, 2, 3, 4, 5] 
output_filename = "code_contests_levels0to12.csv"

print("Loading the 'deepmind/code_contests' dataset (train split)...")
try:
    ds_train = load_dataset("deepmind/code_contests", split="train")
    print("Dataset loaded successfully.")
    print(f"Total examples in the train split: {len(ds_train)}")

except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

collected_samples = {difficulty: [] for difficulty in target_difficulties}

print(f"\nProcessing dataset to collect ALL examples for difficulties: {target_difficulties}...")

# Iterate through the dataset examples
for i, example in enumerate(tqdm(ds_train, desc="Processing dataset")):
    difficulty = example['difficulty']

    if difficulty in target_difficulties:
        collected_samples[difficulty].append(example)

all_collected_examples = []
print("\nSummary of collected samples:")
for difficulty in target_difficulties:
    num_collected = len(collected_samples[difficulty])
    print(f"  Difficulty {difficulty}: {num_collected} examples collected.")
    all_collected_examples.extend(collected_samples[difficulty])

if not all_collected_examples:
    print("\nNo examples were collected for any of the target difficulties. No CSV will be created.")
else:
    print(f"\nCreating a Dataset object from the collected examples ({len(all_collected_examples)} total)...")
    combined_ds = Dataset.from_list(all_collected_examples)

    # Save the combined dataset to a CSV file
    print(f"Saving combined dataset to '{output_filename}'...")
    try:
        combined_ds.to_csv(output_filename, index=False)  # index=False prevents writing the dataset index
        print(f"CSV file '{output_filename}' created successfully.")

        if os.path.exists(output_filename):
            print(f"File size: {os.path.getsize(output_filename)} bytes")

    except Exception as e:
        print(f"Error saving CSV file: {e}")
