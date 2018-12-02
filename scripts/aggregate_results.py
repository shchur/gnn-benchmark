import argparse

import pandas as pd
import pymongo

from gnnbench.util import get_experiment_config

SEARCH_PARAM_COLUMNS = [
    'dropout_prob',
    'input_dropout_prob',
    'coefficient_dropout_prob',
    'learning_rate',
    'weight_decay',
    'hidden_size',
    'alt_opt',
    'r',
    'agg_transform_size',
    'return_prob'
]


def get_metrics_column_names(experiment_config):
    return [f"{mode}.{metric}" for metric in experiment_config["metrics"] for mode in ["train", "val", "test"]]


def fetch_results_as_df(runs, metrics_columns):
    results_list = []
    for record in runs.find():
        try:
            if 'result' in record or 'config' in record:
                config = record['config']
                result = record['result']
                num_training_runs = config['num_training_runs']
                for run_no in range(num_training_runs):

                    row = {'experiment-name': record['experiment']['name'], 'run_no': run_no}
                    row.update(unfold_dict_recursively(config, run_no, num_training_runs))
                    row.update(unfold_dict_recursively(result, run_no, num_training_runs))

                    # make sure all required columns are in row
                    for column in SEARCH_PARAM_COLUMNS + metrics_columns:
                        if column not in row:
                            row[column] = None
                    results_list.append(row)

        except Exception as e:
            print(e, record['experiment']['name'])
            pass

    return pd.DataFrame(results_list)


def unfold_dict_recursively(_dict, run_no, num_training_runs):
    row = {}
    for entry in _dict:
        if type(_dict[entry]) == dict:
            row.update(unfold_dict_recursively(_dict[entry], run_no, num_training_runs))
        else:
            if type(_dict[entry]) == list and len(_dict[entry]) == num_training_runs:
                row[entry] = _dict[entry][run_no]
            else:
                row[entry] = _dict[entry]
    return row


def compute_final_metrics(results_df, metrics_columns):
    group_experiment = results_df.groupby(
        ['experiment-name', 'dataset', 'model_name']
    )[metrics_columns]
    means = group_experiment.mean()
    stddevs = group_experiment.std()
    return means.merge(stddevs, on=['experiment-name', 'dataset', 'model_name'],
                       suffixes=['.mean', '.stddev'])


def evaluate_search(results_df, metrics_columns):
    group_experiment = results_df.groupby(
        ['experiment-name', 'dataset', 'model_name']
    )
    means = group_experiment[metrics_columns].mean()
    stddevs = group_experiment[metrics_columns].std()
    output_table = means.merge(stddevs,
                               on=['experiment-name', 'dataset', 'model_name'],
                               suffixes=['.mean', '.stddev'])
    searched_params = group_experiment[SEARCH_PARAM_COLUMNS].first()
    output_table = output_table.merge(searched_params,
                                      on=['experiment-name', 'dataset', 'model_name'])

    # remove parameters we have not searched for
    output_table = output_table.dropna(axis="columns", how="all")

    # best performing value to the top
    output_table = output_table.sort_values(['model_name', 'dataset', 'test.accuracy.mean'],
                                            ascending=[True, True, False])
    return output_table


def clear_results(runs):
    print("Removing all results from database.")
    runs.delete_many({})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fetch the results of experiments from the database, '
                                                 'aggregate them and save to a CSV file. ')
    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        help='Path to the YAML configuration file for the experiment.')
    parser.add_argument('-o', '--output-prefix',
                        type=str,
                        default='',
                        help='Prefix added to the names of the output files.')
    parser.add_argument('--clear',
                        action='store_true',
                        help='Remove all entries from the results database. Does not affect pending jobs.')
    args = parser.parse_args()

    _experiment_config = get_experiment_config(args.config_file)
    _db_host = _experiment_config['db_host']
    _db_port = _experiment_config['db_port']
    _db_name = _experiment_config['target_db_name']

    # Get results from the database
    client = pymongo.MongoClient(f"mongodb://{_db_host}:{_db_port}/{_db_name}")
    _runs = client[_db_name].runs
    _metrics_columns = get_metrics_column_names(_experiment_config)

    # clear database if demanded
    if args.clear:
        choice = input(f'Are you sure that you want to delete all records in {_db_name}? [y/N] ').lower()
        if choice == 'y' or choice == 'yes':
            clear_results(_runs)
        else:
            print('Aborting.')
            pass
        exit(0)

    # Aggregate results into a DataFrame
    if _experiment_config['experiment_mode'] == 'hyperparameter_search':
        df = fetch_results_as_df(_runs, _metrics_columns)
        if df.empty:
            raise ValueError("The database contains no records.")
        final_metrics = evaluate_search(df, _metrics_columns)
        print(final_metrics.to_string())
    elif _experiment_config['experiment_mode'] == 'fixed_configurations':
        df = fetch_results_as_df(_runs, _metrics_columns)
        if df.empty:
            raise ValueError("The database contains no records.")
        final_metrics = compute_final_metrics(df, _metrics_columns)
        print(final_metrics.to_string())
    else:
        raise ValueError(f"Unsupported experiment mode {_experiment_config['experiment_mode']}")

    # Store results to a csv file
    with open(f"{args.output_prefix}{_experiment_config['experiment_name']}_results_aggregated.csv", 'w') as f:
        final_metrics.to_csv(f)
    with open(f"{args.output_prefix}{_experiment_config['experiment_name']}_results_raw.csv", 'w') as f:
        df.to_csv(f)
