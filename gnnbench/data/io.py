import os
import numpy as np
import scipy.sparse as sp

from gnnbench.data.preprocess import eliminate_self_loops as eliminate_self_loops_adj, largest_connected_components


class SparseGraph:
    """Attributed labeled graph stored in sparse matrix form.

    """
    def __init__(self, adj_matrix, attr_matrix=None, labels=None,
                 node_names=None, attr_names=None, class_names=None, metadata=None):
        """Create an attributed graph.

        Parameters
        ----------
        adj_matrix : sp.csr_matrix, shape [num_nodes, num_nodes]
            Adjacency matrix in CSR format.
        attr_matrix : sp.csr_matrix or np.ndarray, shape [num_nodes, num_attr], optional
            Attribute matrix in CSR or numpy format.
        labels : np.ndarray, shape [num_nodes], optional
            Array, where each entry represents respective node's label(s).
        node_names : np.ndarray, shape [num_nodes], optional
            Names of nodes (as strings).
        attr_names : np.ndarray, shape [num_attr]
            Names of the attributes (as strings).
        class_names : np.ndarray, shape [num_classes], optional
            Names of the class labels (as strings).
        metadata : object
            Additional metadata such as text.

        """
        # Make sure that the dimensions of matrices / arrays all agree
        if sp.isspmatrix(adj_matrix):
            adj_matrix = adj_matrix.tocsr().astype(np.float32)
        else:
            raise ValueError("Adjacency matrix must be in sparse format (got {0} instead)"
                             .format(type(adj_matrix)))

        if adj_matrix.shape[0] != adj_matrix.shape[1]:
            raise ValueError("Dimensions of the adjacency matrix don't agree")

        if attr_matrix is not None:
            if sp.isspmatrix(attr_matrix):
                attr_matrix = attr_matrix.tocsr().astype(np.float32)
            elif isinstance(attr_matrix, np.ndarray):
                attr_matrix = attr_matrix.astype(np.float32)
            else:
                raise ValueError("Attribute matrix must be a sp.spmatrix or a np.ndarray (got {0} instead)"
                                 .format(type(attr_matrix)))

            if attr_matrix.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency and attribute matrices don't agree")

        if labels is not None:
            if labels.shape[0] != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the label vector don't agree")

        if node_names is not None:
            if len(node_names) != adj_matrix.shape[0]:
                raise ValueError("Dimensions of the adjacency matrix and the node names don't agree")

        if attr_names is not None:
            if len(attr_names) != attr_matrix.shape[1]:
                raise ValueError("Dimensions of the attribute matrix and the attribute names don't agree")

        self.adj_matrix = adj_matrix
        self.attr_matrix = attr_matrix
        self.labels = labels
        self.node_names = node_names
        self.attr_names = attr_names
        self.class_names = class_names
        self.metadata = metadata

    def num_nodes(self):
        """Get the number of nodes in the graph."""
        return self.adj_matrix.shape[0]

    def num_edges(self):
        """Get the number of edges in the graph.

        For undirected graphs, (i, j) and (j, i) are counted as single edge.
        """
        if self.is_directed():
            return int(self.adj_matrix.nnz)
        else:
            return int(self.adj_matrix.nnz / 2)

    def get_neighbors(self, idx):
        """Get the indices of neighbors of a given node.

        Parameters
        ----------
        idx : int
            Index of the node whose neighbors are of interest.

        """
        return self.adj_matrix[idx].indices

    def is_directed(self):
        """Check if the graph is directed (adjacency matrix is not symmetric)."""
        return (self.adj_matrix != self.adj_matrix.T).sum() != 0

    def to_undirected(self):
        """Convert to an undirected graph (make adjacency matrix symmetric)."""
        if self.is_weighted():
            raise ValueError("Convert to unweighted graph first.")
        else:
            self.adj_matrix = self.adj_matrix + self.adj_matrix.T
            self.adj_matrix[self.adj_matrix != 0] = 1
        return self

    def is_weighted(self):
        """Check if the graph is weighted (edge weights other than 1)."""
        return np.any(np.unique(self.adj_matrix[self.adj_matrix != 0].A1) != 1)

    def to_unweighted(self):
        """Convert to an unweighted graph (set all edge weights to 1)."""
        self.adj_matrix.data = np.ones_like(self.adj_matrix.data)
        return self

    # Quality of life (shortcuts)
    def standardize(self):
        """Select the LCC of the unweighted/undirected/no-self-loop graph.

        All changes are done inplace.

        """
        G = self.to_unweighted().to_undirected()
        G = eliminate_self_loops(G)
        G = largest_connected_components(G, 1)
        return G

    def unpack(self):
        """Return the (A, X, z) triplet."""
        return self.adj_matrix, self.attr_matrix, self.labels


def eliminate_self_loops(G):
    G.adj_matrix = eliminate_self_loops_adj(G.adj_matrix)
    return G


def load_dataset(data_path):
    """Load a dataset.

    Parameters
    ----------
    name : str
        Name of the dataset to load.

    Returns
    -------
    sparse_graph : SparseGraph
        The requested dataset in sparse format.

    """
    if not data_path.endswith('.npz'):
        data_path += '.npz'
    if os.path.isfile(data_path):
        return load_npz_to_sparse_graph(data_path)
    else:
        raise ValueError(f"{data_path} doesn't exist.")


def load_npz_to_sparse_graph(file_name):
    """Load a SparseGraph from a Numpy binary file.

    Parameters
    ----------
    file_name : str
        Name of the file to load.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph in sparse matrix format.

    """
    with np.load(file_name) as loader:
        loader = dict(loader)
        adj_matrix = sp.csr_matrix((loader['adj_data'], loader['adj_indices'], loader['adj_indptr']),
                                   shape=loader['adj_shape'])

        if 'attr_data' in loader:
            # Attributes are stored as a sparse CSR matrix
            attr_matrix = sp.csr_matrix((loader['attr_data'], loader['attr_indices'], loader['attr_indptr']),
                                        shape=loader['attr_shape'])
        elif 'attr_matrix' in loader:
            # Attributes are stored as a (dense) np.ndarray
            attr_matrix = loader['attr_matrix']
        else:
            attr_matrix = None

        if 'labels_data' in loader:
            # Labels are stored as a CSR matrix
            labels = sp.csr_matrix((loader['labels_data'], loader['labels_indices'], loader['labels_indptr']),
                                   shape=loader['labels_shape'])
        elif 'labels' in loader:
            # Labels are stored as a numpy array
            labels = loader['labels']
        else:
            labels = None

        node_names = loader.get('node_names')
        attr_names = loader.get('attr_names')
        class_names = loader.get('class_names')
        metadata = loader.get('metadata')

    return SparseGraph(adj_matrix, attr_matrix, labels, node_names, attr_names, class_names, metadata)


def save_sparse_graph_to_npz(filepath, sparse_graph):
    """Save a SparseGraph to a Numpy binary file.

    Parameters
    ----------
    filepath : str
        Name of the output file.
    sparse_graph : gust.SparseGraph
        Graph in sparse matrix format.

    """
    data_dict = {
        'adj_data': sparse_graph.adj_matrix.data,
        'adj_indices': sparse_graph.adj_matrix.indices,
        'adj_indptr': sparse_graph.adj_matrix.indptr,
        'adj_shape': sparse_graph.adj_matrix.shape
    }
    if sp.isspmatrix(sparse_graph.attr_matrix):
        data_dict['attr_data'] = sparse_graph.attr_matrix.data
        data_dict['attr_indices'] = sparse_graph.attr_matrix.indices
        data_dict['attr_indptr'] = sparse_graph.attr_matrix.indptr
        data_dict['attr_shape'] = sparse_graph.attr_matrix.shape
    elif isinstance(sparse_graph.attr_matrix, np.ndarray):
        data_dict['attr_matrix'] = sparse_graph.attr_matrix

    if sp.isspmatrix(sparse_graph.labels):
        data_dict['labels_data'] = sparse_graph.labels.data
        data_dict['labels_indices'] = sparse_graph.labels.indices
        data_dict['labels_indptr'] = sparse_graph.labels.indptr
        data_dict['labels_shape'] = sparse_graph.labels.shape
    elif isinstance(sparse_graph.labels, np.ndarray):
        data_dict['labels'] = sparse_graph.labels

    if sparse_graph.node_names is not None:
        data_dict['node_names'] = sparse_graph.node_names

    if sparse_graph.attr_names is not None:
        data_dict['attr_names'] = sparse_graph.attr_names

    if sparse_graph.class_names is not None:
        data_dict['class_names'] = sparse_graph.class_names

    if sparse_graph.metadata is not None:
        data_dict['metadata'] = sparse_graph.metadata

    if not filepath.endswith('.npz'):
        filepath += '.npz'

    np.savez(filepath, **data_dict)

