from transformers import BertModel, BertTokenizer
import numpy as np
import torch
import pandas as pd
import os
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = BertTokenizer.from_pretrained("prot_bert", do_lower_case=False)
model = BertModel.from_pretrained("prot_bert")
model.to(device)
model.eval()

file_path = "raw_data_processed.csv"  
save_dir = "ProtBert_all"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

data = pd.read_csv(file_path)

start_time = time.time()

total_entries = len(data)
for index, row in data.iterrows():
    entry, sequence = row["Entry"], row["Sequence"]
    #print(f"Original entry: {entry}, Sequence: {sequence}")
    
    spaced_sequence = " ".join(sequence)
    #print(f"Spaced Sequence: {spaced_sequence}")
    
    input_ids = tokenizer.encode(spaced_sequence, return_tensors="pt", add_special_tokens=False)
    #print(tokenizer.convert_ids_to_tokens(input_ids))

    input_ids = input_ids.to(device)

    #import pdb; pdb.set_trace()

    with torch.no_grad():
        embedding_repr = model(input_ids=input_ids)
        
        emb = embedding_repr.last_hidden_state[0, :, :]  

        np_representation = emb.cpu().numpy()
        
        print(f"Shape of representation for {entry}: {np_representation.shape}")

        save_path = os.path.join(save_dir, f"{entry}.npy")
        np.save(save_path, np_representation)

    current_time = time.time()
    elapsed_time = (current_time - start_time) / 60
    progress_percentage = ((index + 1) / total_entries) * 100
    print(f"Processed {index + 1}/{total_entries} sequences. Progress: {progress_percentage:.2f}%. Elapsed time: {elapsed_time:.2f} minutes.")

print("Finished processing all sequences.")
