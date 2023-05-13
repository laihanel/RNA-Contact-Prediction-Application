import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import re
from sklearn import preprocessing


def encode_onehot(labels):
    classes = set(labels)
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in
                    enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)),
                             dtype=np.int32)
    return labels_onehot

# read fasta file
def fasta_read(f_name):
    """
    Reads a fasta file and returns a pandas dataframe with rows as IDs and columns as positions in the sequences
    :param f_name: str, path to file
    :return: pandas.DataFrame
    """
    f = open(f_name, 'r')
    lines = f.readlines()
    lines = [line for line in lines if line != '\n']
    id_re = re.compile(r'>(\S+)')
    seq_re = re.compile(r'^(\S+)$')

    tmp = {}

    for line in lines:
        id_h = id_re.search(line)
        if id_h:
            seq_l = None
            seq_id = id_h.group(1)
        else:
            if seq_l is None:
                seq_l = seq_re.search(line).group(1)
            else:
                seq_l = seq_l + seq_re.search(line).group(1)
            tmp[seq_id] = list(seq_l.upper())
    return pd.DataFrame.from_dict(tmp, orient='index')

# read data function
def read_data(file_path, seq_type=None, is_main=True, gap_threshold=0.7):
    """
    Reads data file from tsv, csv, xlsx, xls, fas, fa, and fasta formats and returns a Pandas.DataFrame
    :param file_path: str, path to file
    :param seq_type: str, 'nu' for nucleotides and 'aa' for 'amino-acid' sequences
    :param is_main: bool, True means that this is the MSA file
    :param gap_threshold: float, columns with missing values over the gap_threshold% will be dropped and
     eliminate from the analysis.
    :return: pandas.DataFrame
    """
    if file_path.endswith('.csv'):
        dat = pd.read_csv(file_path, sep=',', index_col=0)
    elif file_path.endswith('.tsv'):
        dat = pd.read_csv(file_path, sep='\t', index_col=0)
    elif file_path.endswith(('.xlsx', '.xls')):
        dat = pd.read_excel(file_path, sep='\t', index_col=0)
    elif any(file_path.endswith(s) for s in ['fasta', 'fas', 'fa']):
        # importing seq data
        dat = fasta_read(f_name=file_path)
    else:
        print('For now, we can only read csv, tsv, excel, and fasta files.')
        exit()

    if is_main:
        # replacing unwanted characters with nan
        if seq_type == 'nu':
            na_values = ['-', 'r', 'y', 'k', 'm', 's', 'w', 'b', 'd', 'h', 'v', 'n']
        else:
            na_values = ['-', 'X', 'B', 'Z', 'J']
        to_replace = []
        for vl in na_values:
            to_replace.append(vl.upper())
        dat.replace(to_replace, np.nan, inplace=True)
        if gap_threshold > 0:
            col_to_drop = dat.columns[dat.isnull().sum() > (gap_threshold * dat.shape[0])]
            dat.drop(col_to_drop, axis=1, inplace=True)

        # naming each position as p + its rank
        dat.columns = [str('p' + str(i)) for i in range(1, dat.shape[1] + 1)]
    return dat
# cite https://github.com/omicsEye/deepbreaks/blob/93fcc2e77dc142694ec3e91afd7bacaaebe52e4c/deepBreaks/preprocessing.py#L33

def matrix_to_array(pred):
    n, m = np.where(pred == 1)
    prediction = list(zip(n.tolist(), m.tolist()))
    pred_list = []
    for (x, y) in prediction:
        pred_list.append([x, y])
    return np.array(pred_list)



def load_data(input_msa, input_meta, input_contact, dataset="hiv"):
    """Load citation network dataset (hiv only for now)"""
    print('Loading {} dataset...'.format(dataset))
    seqFileName = input_msa
    meta_data = input_meta

    df = read_data(seqFileName, seq_type='nu', is_main=True)
    metaData = read_data(meta_data, is_main=False)
    le = preprocessing.LabelEncoder()
    idx_features_labels = df.apply(le.fit_transform)

    features = sp.csr_matrix(idx_features_labels.to_numpy(), dtype=np.float32)
    labels = encode_onehot(metaData["Subtype"])

    # build graphf
    idx = np.arange(0, len(idx_features_labels), dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}
    edges_matrix = pd.read_csv(input_contact, sep=",", header=None)
    pred = edges_matrix.to_numpy().astype(int)
    edges_unordered = matrix_to_array(pred)
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())),
                     dtype=np.int32).reshape(edges_unordered.shape)
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # build symmetric adjacency matrix
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize(features)
    adj = normalize(adj + sp.eye(adj.shape[0]))

    idx_train = range(24500)
    idx_val = range(24501, 29500)
    idx_test = range(29501, 35400)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

if __name__ == "__main__":
    adj, features, labels, idx_train, idx_val, idx_test = load_data("../CoT-RNA-Transfer/data/hiv/hiv_V3_B_C_nu_clean.fasta",
                                                                    "../CoT-RNA-Transfer/data/hiv/results_V3_B_C_meta.csv",
                                                                    "../CoT-RNA-Transfer/outputs/pred.txt")