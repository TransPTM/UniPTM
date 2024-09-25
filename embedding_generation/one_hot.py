import numpy as np
import pandas as pd
import os
import time
import torch

# Amino acid to index mapping
res_to_id = {
    "X": 0, "A": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7, "I": 8, "K": 9,
    "L": 10, "M": 11, "N": 12, "P": 13, "Q": 14, "R": 15, "S": 16, "T": 17, "V": 18,
    "W": 19, "Y": 20, "U": 2, "Z": 0
}

file_path = "raw_data_processed.csv"  # Ensure path is correct
save_dir = "one_hot_all"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data = pd.read_csv(file_path)

start_time = time.time()
total_entries = len(data)

for index, row in data.iterrows():
    entry, sequence = row["Entry"], row["Sequence"]
    
    # Convert sequence to indices
    sequence_indices = [res_to_id.get(res, 21) for res in sequence]  # 21 as a generic index for unknown residues
    # Convert indices to one-hot encoding
    one_hot_matrix = torch.nn.functional.one_hot(torch.tensor(sequence_indices), num_classes=len(res_to_id)+1).numpy()  # +1 for potential unknown residues

    save_path = os.path.join(save_dir, f"{entry}.npy")
    np.save(save_path, one_hot_matrix)

    current_time = time.time()
    elapsed_time = (current_time - start_time) / 60
    progress_percentage = ((index + 1) / total_entries) * 100
    print(f"Processed {index + 1}/{total_entries} sequences. Progress: {progress_percentage:.2f}%. Elapsed time: {elapsed_time:.2f} minutes.")

print("Finished processing all sequences.")
