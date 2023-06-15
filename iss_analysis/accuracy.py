import iss_analysis as iss
import time
import numpy as np
import pandas as pd
import re

def filter_genes(gene_names, gene_list):
    # get rid of gene models etc
    skewed_genes = np.array([gene not in gene_list for gene in gene_names])
    keep_genes = np.logical_not(skewed_genes)

    return keep_genes


def load_data_tasic_2018(datapath, gene_list, gene_classes, filter_neurons=True):
    """
    Load the scRNAseq data from Tasic et al., "Shared and distinct
    transcriptomic cell types across neocortical areas", Nature, 2018.

    Args:
        datapath: path to the data

    Returns:
        exons_matrix: n cells x n genes matrix (numpy.ndarray) of read counts
        cluster_ids: numpy.array of cluster assignments from the cell metadata
        cluster_means: n clusters x n genes matrix (numpy.ndarray) of mean read
            counts for each cluster
        cluster_labels: list of cluster names
        gene_names: pandas.Series of gene names

    """
    fname_metadata = f'{datapath}mouse_VISp_2018-06-14_samples-columns.csv'
    metadata = pd.read_csv(fname_metadata, low_memory=False)
    fname = f'{datapath}mouse_VISp_2018-06-14_exon-matrix.csv'
    exons = pd.read_csv(fname, low_memory=False)
    fname_genes = f'{datapath}mouse_VISp_2018-06-14_genes-rows.csv'
    genes = pd.read_csv(fname_genes, low_memory=False)
    gene_names = genes['gene_symbol']
    metadata.set_index('sample_name', inplace=True)
    keep_genes = filter_genes(gene_names, gene_list)
    exons = exons.iloc[keep_genes]
    gene_names = gene_names.iloc[keep_genes]
    exons_df = metadata.join(exons.T, on='sample_name')
    # only include neurons
    if gene_classes == 'glu':
        include_classes = [
            'Glutamatergic',
            'GABAergic',
        ]
    else:
        include_classes = [
            'GABAergic',
        ]
    # get rid of low quality cells etc
    if filter_neurons:
        exons_subset = exons_df[exons_df['class'].isin(include_classes)]
    else:
        exons_subset = exons_df
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('ALM')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Doublet')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Batch')]
    exons_subset = exons_subset[~exons_subset['cluster'].str.contains('Low Quality')]
    exons_subset = exons_subset[~exons_subset['subclass'].str.contains('High Intronic')]

    return exons_subset, gene_names

def optimize_gene_set(cluster_probs, cluster_ids, gene_names, gene_set=(),
                      niter=150, subsample_cells=1):
    gene_set_history = []
    accuracy_history = []
    ordered_gene_list = []
    include_genes = np.isin(np.array(gene_names), gene_set)
    for i in range(niter):
        if subsample_cells < 1:
            cell_idx = np.random.rand(cluster_probs.shape[0]) < subsample_cells
            b, accuracy = iss.pick_genes.next_best_gene(include_genes, cluster_probs[cell_idx,:,:], cluster_ids[cell_idx])
        else:
            b, accuracy = iss.pick_genes.next_best_gene(include_genes, cluster_probs, cluster_ids)

        include_genes[b] = True
        print(f'added {gene_names.iloc[b]}, accuracy = {accuracy}')
        ordered_gene_list.append(gene_names.iloc[b])
        
        gene_set_history.append(include_genes)
        accuracy_history.append(accuracy)
    return include_genes, gene_set_history, accuracy_history, ordered_gene_list

def determine_accuracy(gene_list, parameters):
    
    exons_df_skewed, gene_names_skewed = load_data_tasic_2018('/camp/lab/znamenskiyp/home/shared/resources/allen2018/', gene_list, gene_classes = 'glu')

    #split into test and train groups
    train_skewed, test_skewed, cluster_labels_skewed = iss.pick_genes.train_test_split(exons_df_skewed, 'cluster', gene_filter='\d', efficiency =0.001)

    #compute mean expression for each cluster and gene
    exons_matrix_skewed, cluster_ids_skewed, cluster_means_skewed, cluster_labels_skewed = iss.pick_genes.compute_means(exons_df_skewed, 'cluster', gene_filter='\d')

    resampled_exons_matrix_skewed, resampled_cluster_means_skewed = iss.pick_genes.resample_counts(exons_matrix_skewed, cluster_means_skewed, efficiency=0.001)

    #precompute cluster probabilities
    cluster_probs_skewed = iss.pick_genes.compute_cluster_probabilities(resampled_exons_matrix_skewed, resampled_cluster_means_skewed, nu=0.001)


    #pick genes in order from unordered predefined set
    include_genes_skewed, gene_set_history_skewed, accuracy_history_skewed, ordered_gene_list_skewed = optimize_gene_set(cluster_probs_skewed, cluster_ids_skewed, gene_names_skewed, subsample_cells=1)

    accuracy_train, accuracy_test = iss.pick_genes.evaluate_gene_set(train_skewed, test_skewed, ordered_gene_list_skewed, gene_names_skewed)

    np.save(f'/camp/lab/znamenskiyp/home/users/becalia/data/V1_padlocks/results/accuracy_{parameters}.npy', accuracy_test, allow_pickle=True)