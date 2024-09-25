import pandas as pd

def csv_to_fasta_with_mapping(df, fasta_file_path, mapping_file_path):
    with open(fasta_file_path, 'w') as fasta_file, open(mapping_file_path, 'w') as mapping_file:
     
        mapping_file.write("Entry,Modified residue,Position,Label,Mask,Pos_num,TargetAA_num\n")
        for _, row in df.iterrows():
       
            description = f">{row['Entry']}"
            sequence = row['Sequence']
        
            fasta_file.write(f"{description}\n{sequence}\n\n")
       
            mapping_info = f"{row['Entry']},{row['Modified residue']},{row['Position']},{row['Label']},{row['Mask']},{row['Pos_num']},{row['TargetAA_num']}\n"
            mapping_file.write(mapping_info)

fasta_file_path = './phosphotyrosine.fasta'
mapping_file_path = './phosphotyrosine_mapping.csv'


df = pd.read_csv('./phosphotyrosine.csv')

csv_to_fasta_with_mapping(df, fasta_file_path, mapping_file_path)
