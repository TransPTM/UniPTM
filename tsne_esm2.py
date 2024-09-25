import time
import torch
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')  # Set the backend for non-display environment
import matplotlib.pyplot as plt
from Data.data_processing import ProteinDataset, collate_fn  # Ensure these are correctly imported

def get_site_features(batch):
    """
    Extract features, labels, and mask from batch data.
    Assume features are stored under the key 'embedding', labels under 'label', and mask under 'mask'.
    """
    check_dataset_keys(batch, keys=['embedding', 'label', 'mask'])
    emb = batch['embedding']
    label = batch['label']
    mask = batch['mask'].bool()  # Ensure mask is boolean

    features = emb[mask]
    labels = label[mask]

    return features, labels

def check_dataset_keys(batch, keys):
    """
    Check if the required keys exist in the batch to prevent runtime errors.
    """
    missing_keys = [key for key in keys if key not in batch]
    if missing_keys:
        raise KeyError(f"Missing keys in the batch: {', '.join(missing_keys)}")

def main():
    """
    Main function to execute data loading, processing, and visualization.
    """
    start_time = time.time()

    # Loading the data
    EMB_PATH = './Data/esm2_all'
    data_path = '/home/menglingkuan/proj/Data/N6,N6-dimethyllysine_clustered_splited.csv'
    data = pd.read_csv(data_path)
    train_data = data[data['Set'] == 'train']

    train_dataset = ProteinDataset(train_data, EMB_PATH, target_aa='K')
    train_dataloader = DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    all_features = []
    all_labels = []

    # Extract features and labels directly from the dataset
    for batch in train_dataloader:
        site_features, site_labels = get_site_features(batch)
        for i in range(site_features.shape[0]):
            feature = site_features[i].detach().numpy()
            label = site_labels[i].item()
            all_features.append(feature)
            all_labels.append(label)

    all_features = np.array(all_features)
    all_labels = np.array(all_labels)

    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42)  # 2D t-SNE representation
    transformed_features = tsne.fit_transform(all_features)

    # Visualization
    plt.figure(figsize=(8, 8))
    colors = {1: 'black', 0: 'grey'}  # Define custom colors for plot
    point_size = 5  # Size of points in the plot

    for label in np.unique(all_labels):
        indices = all_labels == label
        plt.scatter(transformed_features[indices, 0], transformed_features[indices, 1],
                    color=colors[label],
                    label=f"Dimethylation (K)" if label == 1 else "Non-Dimethylation (K)",
                    s=point_size)


    # Enhance the legend
    legend = plt.legend(markerscale=2, scatterpoints=1, loc='lower left')   # Increase legend point size
    for handle in legend.legendHandles:
        handle.set_sizes([30.0])  # Increase the size of the legend markers



    plt.xlabel('t-SNE 1' , fontsize=15, weight='bold')
    plt.ylabel('t-SNE 2' , fontsize=15, weight='bold')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.gca().spines['top'].set_linewidth(2)
    plt.gca().spines['right'].set_linewidth(2)
    plt.gca().spines['bottom'].set_linewidth(2)
    plt.gca().spines['left'].set_linewidth(2)

    plt.savefig('tSNE_esm2_visualization.png')
    plt.savefig('tSNE_esm2_visualization.svg')
    plt.show()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds")

if __name__ == '__main__':
    main()
