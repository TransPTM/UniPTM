import torch
import esm
print(esm.__file__)
import pandas as pd
import os
import numpy as np

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cpu")  
else:
    device = torch.device("cpu")

# Load ESM-2 model and move it to the GPU if available
model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
model.to(device)
batch_converter = alphabet.get_batch_converter()
model.eval()  # disables dropout for deterministic results

file_path = "/proj/Data/raw_data_processed.csv"
save_dir = "/proj/Data/esm2_all_256"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data = pd.read_csv(file_path)
total = data.shape[0]  

for index, row in enumerate(data.iterrows()):  
    entry = row[1]["Entry"]  
    sequence = row[1]["Sequence"]
    formatted_data = [(entry, sequence)]

    batch_labels, batch_strs, batch_tokens = batch_converter(formatted_data)

    # Move batch_tokens to the GPU
    batch_tokens = batch_tokens.to(device)

    with torch.no_grad():
        results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        token_representation = results["representations"][33]

    save_path = os.path.join(save_dir, f"{entry}.npy")
    np_representation = token_representation[:, 1:-1, :].squeeze(0).cpu().numpy()
    np.save(save_path, np_representation)
    
    # 打印当前进度
    print(f"Progress: {index + 1}/{total} ({(index + 1) / total * 100:.2f}%)")
    print(np_representation.shape)
