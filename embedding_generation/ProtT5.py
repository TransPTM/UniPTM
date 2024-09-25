from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
import re
import torch
import pandas as pd
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = T5Tokenizer.from_pretrained("prot/prot_t5_xl_uniref50", do_lower_case=False)
model = T5EncoderModel.from_pretrained("prot/prot_t5_xl_uniref50")
model.to(device)
model.eval()  # Set the model to inference mode

file_path = "raw_data_processed.csv"
save_dir = "T5_all"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data = pd.read_csv(file_path)

start_time = time.time()

total_entries = len(data)
for index, row in enumerate(data.iterrows()):
    entry, sequence = row[1]["Entry"], row[1]["Sequence"]

    sequence = " ".join(list(re.sub(r"[UZOB]", "X", sequence)))
    input_ids = tokenizer.encode(sequence, return_tensors="pt", add_special_tokens=True)
    input_ids = input_ids.to(device)

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids)
        emb = embedding_repr.last_hidden_state[0, :-1, :]  # Only remove the embedding for the last special token
        np_representation = emb.cpu().numpy()

        print(f"Shape of representation for {entry}: {np_representation.shape}")

        save_path = os.path.join(save_dir, f"{entry}.npy")
        np.save(save_path, np_representation)


    current_time = time.time()
    elapsed_time = (current_time - start_time) / 60  


    progress_percentage = ((index + 1) / total_entries) * 100
    print(f"Processed {index + 1}/{total_entries} sequences. Progress: {progress_percentage:.2f}%. Elapsed time: {elapsed_time:.2f} minutes.")

print("Finished processing all sequences.")
