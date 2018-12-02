import numpy as np
import scipy.sparse as sp

from collections import Counter

from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer, normalize


def to_binary_bag_of_words(features):
    """Converts TF/IDF features to binary bag-of-words features."""
    features_copy = features.tocsr()
    features_copy.data[:] = 1.0
    return features_copy


def normalize_adj(A):
    """Compute D^-1/2 * A * D^-1/2."""
    # Make sure that there are no self-loops
    A = eliminate_self_loops(A)
    D = np.ravel(A.sum(1))
    D[D == 0] = 1  # avoid division by 0 error
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def renormalize_adj(A):
    """Renormalize the adjacency matrix (as in the GCN paper)."""
    A_tilde = A.tolil()
    A_tilde.setdiag(1)
    A_tilde = A_tilde.tocsr()
    A_tilde.eliminate_zeros()
    D = np.ravel(A.sum(1))
    D_sqrt = np.sqrt(D)
    return A / D_sqrt[:, None] / D_sqrt[None, :]


def row_normalize(matrix):
    """Normalize the matrix so that the rows sum up to 1."""
    return normalize(matrix, norm='l1', axis=1)


def add_self_loops(A, value=1.0):
    """Set the diagonal."""
    A = A.tolil()  # make sure we work on a copy of the original matrix
    A.setdiag(value)
    A = A.tocsr()
    if value == 0:
        A.eliminate_zeros()
    return A


def eliminate_self_loops(A):
    """Remove self-loops from the adjacency matrix."""
    A = A.tolil()
    A.setdiag(0)
    A = A.tocsr()
    A.eliminate_zeros()
    return A


def largest_connected_components(sparse_graph, n_components=1):
    """Select the largest connected components in the graph.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    n_components : int, default 1
        Number of largest connected components to keep.

    Returns
    -------
    sparse_graph : SparseGraph
        Subgraph of the input graph where only the nodes in largest n_components are kept.

    """
    _, component_indices = sp.csgraph.connected_components(sparse_graph.adj_matrix)
    component_sizes = np.bincount(component_indices)
    components_to_keep = np.argsort(component_sizes)[::-1][:n_components]  # reverse order to sort descending
    nodes_to_keep = [
        idx for (idx, component) in enumerate(component_indices) if component in components_to_keep
    ]
    return create_subgraph(sparse_graph, nodes_to_keep=nodes_to_keep)


def create_subgraph(sparse_graph, _sentinel=None, nodes_to_remove=None, nodes_to_keep=None):
    """Create a graph with the specified subset of nodes.

    Exactly one of (nodes_to_remove, nodes_to_keep) should be provided, while the other stays None.
    Note that to avoid confusion, it is required to pass node indices as named arguments to this function.

    Parameters
    ----------
    sparse_graph : SparseGraph
        Input graph.
    _sentinel : None
        Internal, to prevent passing positional arguments. Do not use.
    nodes_to_remove : array-like of int
        Indices of nodes that have to removed.
    nodes_to_keep : array-like of int
        Indices of nodes that have to be kept.

    Returns
    -------
    sparse_graph : SparseGraph
        Graph with specified nodes removed.

    """
    # Check that arguments are passed correctly
    if _sentinel is not None:
        raise ValueError("Only call `create_subgraph` with named arguments',"
                         " (nodes_to_remove=...) or (nodes_to_keep=...)")
    if nodes_to_remove is None and nodes_to_keep is None:
        raise ValueError("Either nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None and nodes_to_keep is not None:
        raise ValueError("Only one of nodes_to_remove or nodes_to_keep must be provided.")
    elif nodes_to_remove is not None:
        nodes_to_keep = [i for i in range(sparse_graph.num_nodes()) if i not in nodes_to_remove]
    elif nodes_to_keep is not None:
        nodes_to_keep = sorted(nodes_to_keep)
    else:
        raise RuntimeError("This should never happen.")

    sparse_graph.adj_matrix = sparse_graph.adj_matrix[nodes_to_keep][:, nodes_to_keep]
    if sparse_graph.attr_matrix is not None:
        sparse_graph.attr_matrix = sparse_graph.attr_matrix[nodes_to_keep]
    if sparse_graph.labels is not None:
        sparse_graph.labels = sparse_graph.labels[nodes_to_keep]
    if sparse_graph.node_names is not None:
        sparse_graph.node_names = sparse_graph.node_names[nodes_to_keep]
    return sparse_graph


def binarize_labels(labels, sparse_output=False, return_classes=False):
    """Convert labels vector to a binary label matrix.

    In the default single-label case, labels look like
    labels = [y1, y2, y3, ...].
    Also supports the multi-label format.
    In this case, labels should look something like
    labels = [[y11, y12], [y21, y22, y23], [y31], ...].

    Parameters
    ----------
    labels : array-like, shape [num_samples]
        Array of node labels in categorical single- or multi-label format.
    sparse_output : bool, default False
        Whether return the label_matrix in CSR format.
    return_classes : bool, default False
        Whether return the classes corresponding to the columns of the label matrix.

    Returns
    -------
    label_matrix : np.ndarray or sp.csr_matrix, shape [num_samples, num_classes]
        Binary matrix of class labels.
        num_classes = number of unique values in "labels" array.
        label_matrix[i, k] = 1 <=> node i belongs to class k.
    classes : np.array, shape [num_classes], optional
        Classes that correspond to each column of the label_matrix.

    """
    if hasattr(labels[0], '__iter__'):  # labels[0] is iterable <=> multilabel format
        binarizer = MultiLabelBinarizer(sparse_output=sparse_output)
    else:
        binarizer = LabelBinarizer(sparse_output=sparse_output)
    label_matrix = binarizer.fit_transform(labels).astype(np.float32)
    return (label_matrix, binarizer.classes_) if return_classes else label_matrix


def remove_underrepresented_classes(g, train_examples_per_class, val_examples_per_class):
    """Remove nodes from graph that correspond to a class of which there are less than
    num_classes * train_examples_per_class + num_classes * val_examples_per_class nodes.

    Those classes would otherwise break the training procedure.
    """
    min_examples_per_class = train_examples_per_class + val_examples_per_class
    examples_counter = Counter(g.labels)
    keep_classes = set(class_ for class_, count in examples_counter.items() if count > min_examples_per_class)
    keep_indices = [i for i in range(len(g.labels)) if g.labels[i] in keep_classes]

    return create_subgraph(g, nodes_to_keep=keep_indices)
