# GNN-benchmark

This library provides a unified test bench for evaluating graph neural network (GNN) models on the transductive node classification task.
The framework provides a simple interface for running different models on several datasets while using multiple train/validation/test splits.
In addition, the framework allows to automatically perform hyperparameter tuning for all the models using random search.

This framework uses [Sacred] as a backend for keeping track of experimental results, and all the GNN models are implemented in [TensorFlow].
The current version only supports training models on GPUs.
This package was tested on Ubuntu 16.04 LTS with Python 3.6.6.

[TensorFlow]: https://www.tensorflow.org
[Sacred]: https://github.com/IDSIA/sacred


## Table of contents
1. [Installation](#installation)
2. [Running experiments](#running-experiments-with-gnnbench)
   1. [General structure](#general-structure)
    2. [Configuring experiments](#configuring-experiments)
    3. [Creating jobs](#creating-jobs)
    4. [Running jobs](#running-jobs)
    5. [Retrieving and aggregating results](#retrieving-and-aggregating-the-results)
    6. [Cleaning up the database](#cleaning-up-the-database)
3. [GNN models](#gnn-models)
4. [Datasets](#datasets)
5. [Extending the framework](#extending-the-framework)
    1. [Adding new models](#adding-new-models)
    2. [Adding new datasets](#adding-new-datasets)
    3. [Adding new metrics](#adding-new-metrics)
6. [Cite](#cite)

## Installation
1. Install [MongoDB]. The implementation was tested with MongoDB `3.6.4`.

    This framework will automatically create a database called `pending` and databases with the
    names provided in the [experiment configuration files](#configuring-experiments).
    Make sure these databases do not exist or are empty before running your experiments.

2. Install Python dependencies from the [requirements.txt](requirements.txt) file.
When using [conda] this can be done as
    ```bash
    cd gnn-benchmark/
    while read requirement; do conda install --yes $requirement; done < requirements.txt
    ```
    All required packages except [Sacred] are available via conda.
    Sacred can be installed with
    ```bash
    pip install sacred==0.7.3
    ```


3. Install the `gnnbench` package
    ```bash
    pip install -e .  # has to be run in the directory with setup.py file, i.e. in gnn-benchmark/
    ```

[MongoDB]: https://www.mongodb.com
[conda]: https://conda.io/docs

## Running experiments with `gnnbench`
### General structure
Performing experiments with `gnnbench` consists of four steps:
1. [**Define configuration**](#configuring-experiments) of the experiments using YAML files.

2. [**Create jobs.**](#creating-jobs)
Based on the configuration files defined in the previous step,
a list of jobs to be performed is created and saved to the database.
Each job is represented as a record in the MongoDB database.
3. [**Spawn worker threads.**](#running-jobs)
Each thread retrieves one job from the database at a time and runs it.
4. [**Retrieve results.**](#retrieving-and-aggregating-the-results)
The results are retrieved from the database, aggregated and stored in CSV format.


### Configuring experiments
The framework supports two types of experiments:
* **Fixed configurations:**
All models have a fixed predefined configuration (i.e. hyperparameter settings).
Each model is run on each dataset for the given number of train/validation/test splits
and random weight initializations.
See [config/fixed_config.conf.yaml](config/fixed_configs.conf.yaml) for an example.
Default base configuration shared across all models is located in [config/train.conf.yaml](config/train.conf.yaml).

    The fixed model-specific configuration for each model is defined in a respective YAML file
    (see [configs/optimized/gcn.conf.yaml](config/optimized/gcn.conf.yaml) for an example)
    and may override the defaults from [config/train.conf.yaml](config/train.conf.yaml).
    We provide two types of configuration files.
    1. Configurations corresponding to parameters used in the reference implementations are provided in [config/reference/](config/reference/).
    2. Optimized configurations that we found using hyperparameter search are provided in the folder [config/optimized/](config/optimized/).

* **Hyperparameter search:**
For each model, we define a search space over which random hyperparameter search is performed.
Each model is run on each dataset for the given number of train/validation/test splits
and random weight initializations.
See [hyperparameter_search.conf.yaml](config/hyperparameter_search.conf.yaml) for an example.

    Like before, default configurations for all models are provided in [config/train.conf.yaml](config/train.conf.yaml).
    Model-specific parameters (e.g. from [configs/optimized/gcn.conf.yaml](config/optimized/gcn.conf.yaml)) override the default configurations.
    Finally, parameters sampled from the search space override the model specific parameters.
    To summarize, ordered from the highest priority to the lowest priority:
    1. Search parameters (sampled from e.g. [config/searchspace/gcn.searchspace.yaml](config/searchspace/gcn.searchspace.yaml)).
    2. Model-specific parameters (e.g. from [config/optimized/gcn.conf.yaml](config/optimized/gcn.conf.yaml)).
    3. Default parameters (from [config/train.conf.yaml](config/train.conf.yaml)).



### Creating jobs
Use the [scripts/create_jobs.py](scripts/create_jobs.py) script to generate jobs (represented by records in the `pending` database) based on the YAML configuration file.
The script should be called as
```bash
python create_jobs.py -c CONFIG_FILE --op {fixed,search,status,clear,reset}
```
The `-c CONFIG_FILE` argument contains the path to the YAML file defining the experiment.

You can perform different operations by passing different options to the `--op` argument.
- `--op fixed` generates jobs for a **fixed configurations** experiment. Example usage:
```bash
python scripts/create_jobs.py -c config/fixed_configs.conf.yaml --op fixed
```
- `--op search` generates jobs for a **hyperparameter search** experiment. Example usage:
```bash
python scripts/create_jobs.py -c config/hyperparameter_search.conf.yaml --op search
```
- `--op status` displays the status of the jobs in the `pending` database
- `--op reset` allows to reset the running status of all jobs. This is necessary in case some jobs crashed and need to be restarted.
- `--op clear` removes all the pending jobs from the database.


### Running jobs
You can run jobs by spawning worker threads with [scripts/spawn_worker.py](scripts/spawn_worker.py).
The script works by retrieving pending jobs (i.e. records) from the `pending` database and executing them in a subprocess.
Example usage
```bash
python scripts/spawn_worker.py -c configs/fixed_configs.conf.yaml --gpu 0
```
You can run experiments on multiple GPUs in parallel by spawning multiple workers (e.g. using separate [tmux](https://github.com/tmux/tmux) sessions or panes) and passing different values for the `--gpu` parameter.
In theory, it should be possible to run multiple workers on a single GPU, but we haven't tested that and cannot guarantee that it will work.

### Retrieving and aggregating the results
Use the [scripts/aggregate_results.py](scripts/aggregate_results.py) script to retrieve results from the database and aggregate them.
The script takes the following command line arguments:
- `-c CONFIG_FILE` - path to the YAML configuration file defining the experiment of which you want
to retrieve the results.
- `-o OUTPUT_PREFIX` - prefix for the generated CSV files.
The script generates two files:
    - `<OUTPUT_PREFIX><experiment_name>_results_raw.csv` contains the results for each single run of each model, dataset, split and initialization.
    - `<OUTPUT_PREFIX><experiment_name>_results_aggregated.csv` contains the results for each model and dataset averaged over all splits and initializations.
- `--clear` - delete the results of all completed runs from the database and exit.

Example usage:
```bash
python scripts/aggregate_results.py -c configs/fixed_configs.conf.yaml -o results/
```

### Cleaning up the database
If you want to clean up the database, you should run the following commands.
You should replace `CONFIG_FILE` with the path to the YAML config of the experiment that you are running.
1. To stop all running experiments, simply kill all the running `spawn_worker.py` processes.
2. Reset the `running` status of all experiments to `False`
    ```
    python scripts/create_jobs.py -c CONFIG_FILE --reset
    ```
3. Delete all `pending` jobs from the database
    ```
    python scripts/create_jobs.py -c CONFIG_FILE --clear
    ```
4. Delete all finished jobs from the database
    ```
    python scripts/aggregate_results.py -c CONFIG_FILE --clear
    ```



## GNN models
The framework contains the implementations of the following models (located in [gnnbench/models/](gnnbench/models/) directory)
- Graph Convolutional Networks (GCN, [Kipf and Welling, ICLR 2017][1])
- Mixture Model Networks (MoNet, [Monti et al., CVPR 2016][2])
- Graph Attention Networks (GAT, [Velickovic et al., ICLR 2018][3])
- Graph Sampling and Aggregation (GraphSAGE, [Hamilton et al., NIPS 2017][4])
- Baseline models: Multilayer Perceptron, Logistic Regression and Label Propagation.

## Datasets
Following attributed graph datasets are currently included (located in the [gnnbench/data/](gnnbench/data/) directory)
- CORA and CiteSeer ([Sen et al., 2008][5])
- PubMed ([Namata et al., 2012][6])
- CORA-full ([Bojchevski and Günnemann, 2017][7])
- CoAuthor CS and CoAuthor Physics (generated from the [Microsoft Academic Graph dataset](https://kddcup2016.azurewebsites.net/))
- Amazon Computers and Amazon Photo (based on the Amazon dataset from [McAuley et al., 2015][8])

Each graph (dataset) is represented as an N x N adjacency matrix **A**, an N x D attribute matrix **X**, and a vector of node labels **y** of length N.
We store the datasets as `npz` archives. See [gnnbench/data/io.py](gnnbench/data/io.py) and [Adding new datasets](#adding-new-datasets) for information about reading and saving data in this format.

[1]: https://arxiv.org/abs/1609.02907
[2]: http://openaccess.thecvf.com/content_cvpr_2017/papers/Monti_Geometric_Deep_Learning_CVPR_2017_paper.pdf
[3]: https://arxiv.org/abs/1710.10903
[4]: https://arxiv.org/abs/1706.02216
[5]: https://www.aaai.org/ojs/index.php/aimagazine/article/view/2157
[6]: https://dtai.cs.kuleuven.be/events/mlg2012/papers/11_querying_namata.pdf
[7]: https://arxiv.org/abs/1707.03815
[8]: https://arxiv.org/abs/1506.04757

## Extending the framework
You can extend this framework by adding your own models, datasets and metrics functions.

### Adding new models
All models need to extend the `GNNModel` class defined in [gnnbench/models/base_model.py](gnnbench/models/base_model.py).

Each model needs to be defined in a separate file.

In order for the framework to instantiate the model correctly,
you also need to define a *Sacred Ingredient* for the model,
which in general looks as follows:
```python
MODEL_INGREDIENT = Ingredient('model')
@MODEL_INGREDIENT.capture
def build_model(graph_adj, node_features, labels, dataset_indices_placeholder,
                train_feed, trainval_feed, val_feed, test_feed,
                dropout_prob,
                model_specific_param1, model_specific_param2, ...):
    # needed if the model uses dropout
    dropout = tf.placeholder(dtype=tf.float32, shape=[])
    train_feed[dropout] = dropout_prob
    trainval_feed[dropout] = False
    val_feed[dropout] = False
    test_feed[dropout] = False

    return MyModel(node_features, graph_adj, labels, dataset_indices_placeholder,
               dropout_prob=dropout,
               model_specific_param1=model_specific_param1,
               model_specific_param2=model_specific_param2,
               ...)
```
The parameters coming after `test_feed` can then be configured in the model config file
`mymodel.conf.yaml`. In this config file the parameter `model_name` must be the same as the file
name in which the model is defined (case-insensitive).
Have a look at the existing implementations (e.g. [GCN](gnnbench/models/gcn.py)) for an example of
this method.

To run experiments with the new model, create the YAML configuration file for the model
(e.g. `config/mymodel.conf.yaml`).
Then, add this file to the list of models to the experiment config YAML file
```yaml
models:
    - ...
    - "config/mymodel.conf.yaml"
```
and follow the [instructions for running experiments](#running-experiments-with-`gnnbench`).


### Adding new datasets
To add a new dataset, convert your data to the [SparseGraph](gnnbench/data/io.py)
format and save it to an `npz` file
```python
from gnnbench.data.io import SparseGraph, save_sparse_graph_to_npz

# Load the adjacency matrix A, attribute matrix X and labels vector y
# A - scipy.sparse.csr_matrix of shape [num_nodes, num_nodes]
# X - scipy.sparse.csr_matrix or np.ndarray of shape [num_nodes, num_attributes]
# y - np.ndarray of shape [num_nodes]
...

mydataset = SparseGraph(adj_matrix=A, attr_matrix=X, labels=y)
save_sparse_graph_to_npz('path/to/mydataset.npz', mydataset)
```
To run experiments on the new dataset, add the dataset to the YAML configuration of the experiment:
```yaml
datasets:
    - ...
    - "path/to/mydataset.npz"
```


### Adding new metrics
To add a new metric add a new function to the [metrics.py](gnnbench/metrics.py) file. The function
must have the following signature:
```python
def new_metric(ground_truth, predictions):
    """Description of the new amazing metric.

    Parameters
    ----------
    ground_truth : np.ndarray, shape [num_samples]
        True labels.
    predicted : np.ndarray, shape [num_samples]
        Predicted labels.

    Returns
    -------
    score : float
        Value of metric for the given predictions.
    """
```
Then add the metric to the YAML config file for the experiment
```yaml
metrics:
    - ...
    - "new_metric"
```


## Cite
Please cite our paper if you use this code or the newly introduced datasets in your own work:

```bibtex
@article{shchur2018pitfalls,
  title={Pitfalls of Graph Neural Network Evaluation},
  author={Shchur, Oleksandr and Mumme, Maximilian and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan},
  journal={Relational Representation Learning Workshop, NeurIPS 2018},
  year={2018}
}
```
