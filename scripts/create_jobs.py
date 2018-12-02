import argparse

import numpy as np
import yaml

from gnnbench.util import get_pending_collection, generate_random_parameter_settings, get_experiment_config


def generate_configs(pending, fixed, experiment_config):
    with open(experiment_config["default_config"]) as conf:
        train_config = yaml.load(conf)

    for model_config_path in experiment_config['models']:
        with open(model_config_path) as conf:
            model_config = yaml.load(conf)

        if fixed:
            param_sweep = [(experiment_config['experiment_name'], {})]
        else:
            num_different_configs = experiment_config['num_different_configs']
            search_spaces_dict = load_search_config(experiment_config['searchspaces'][model_config['model_name']])
            random_parameter_settings = generate_random_parameter_settings(
                search_spaces_dict, num_different_configs, experiment_config['seed'])

            param_sweep = []
            for i in range(num_different_configs):
                setting = {param: random_parameter_settings[param][i] for param in random_parameter_settings}
                param_sweep.append((f"{experiment_config['experiment_name']}-search{i}", setting))

        insert_configs(pending, param_sweep,
                       train_config=train_config, model_config=model_config, experiment_config=experiment_config)


def add_to_full_config(config_update, train_config, model_config):
    for param in config_update:
        if param in train_config:
            train_config[param] = config_update[param]
        # IMPORTANT! Model config after train config in order to correctly override parameters
        if param in model_config:
            model_config[param] = config_update[param]
    return train_config, model_config


def generate_multiple_splits(num_different_splits, experiment_name, model_name, dataset_path, train_config, model_config,
                             experiment_config):
    global_random_state = np.random.RandomState(experiment_config['seed'])
    experiment_seeds = global_random_state.randint(0, 1000000, num_different_splits)

    return [{
        "running": False,  # flag needed for status tracking of splits
        "config": {
            "experiment_name": experiment_name,
            "model_name": model_name,
            "dataset": dataset_path.split('/')[-1].split('.')[0],
            "num_training_runs": experiment_config["num_inits"],
            "dataset_source": experiment_config["dataset_format"],
            "data_path": dataset_path,
            "split_no": split_no,
            "seed": int(experiment_seeds[split_no]),
            "train_config": train_config,
            "model_config": model_config,
            "target_db_name": experiment_config["target_db_name"],
            "metrics": experiment_config["metrics"]
        }

    } for split_no in range(num_different_splits)]


def insert_configs(pending, param_sweep, train_config, model_config, experiment_config):
    splits = []
    for dataset_path in experiment_config['datasets']:
        for experiment_name, config in param_sweep:
            train_config, model_config = add_to_full_config(config, train_config, model_config)
            splits += generate_multiple_splits(experiment_config['num_different_splits'], experiment_name,
                                               model_config["model_name"], dataset_path,
                                               train_config, model_config, experiment_config)

    print(f"Inserting {len(splits)} configs for model {model_config['model_name']} into pending list.")
    pending.insert_many(splits)
    print("Done inserting configs.")


def load_search_config(searchspace_path):
    with open(searchspace_path, "r") as f:
        return yaml.load(f)


def report_pending_status(pending):
    count = pending.count()
    running = pending.find({"running": True})
    print(f"{count} entries in database, {running.count()} running.")


def reset_running_status(pending):
    print("Setting 'running' to False for all configs in database.")

    if pending.count() <= 0:
        print("No pending jobs. Exiting...")
        return

    pending.update_many({}, {"$set": {"running": False}}, upsert=False)


def clear_pending_configs(pending):
    print("Removing all pending configurations from database.")
    pending.delete_many({})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create jobs for the given experiment. '
                                     'Each job is represented as a record in the "pending" database. '
                                     'See README.md for more details.',
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('--op',
                        required=True,
                        choices=['fixed', 'search', 'status', 'clear', 'reset'],
                        help='Operation to execute.\n'
                             ' - fixed  - Each model is run with a fixed predefined configuration.\n'
                             ' - search - Perform random hyperparameter search for each model in the specified ranges.\n'
                             ' - status - Report status of the currently queued jobs in the "pending" database.\n'
                             ' - clear  - Remove all pending jobs from the database.\n'
                             " - reset  - Reset the status of running jobs in the database. "
                             "It's necessary to continue an interrupted experiment "
                             "(e.g. in case of a machine failure).")
    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        help='Path to the YAML configuration file for the experiment.')
    args = parser.parse_args()

    _experiment_config = get_experiment_config(args.config_file)
    _pending = get_pending_collection(_experiment_config['db_host'], _experiment_config['db_port'])

    if args.op == "fixed":
        if _experiment_config['experiment_mode'] != 'fixed_configurations':
            raise ValueError(f'The "experiment_mode" must be set to "fixed_configurations"'
                             'in {args.config_file} when using the "--op fixed" option')
        generate_configs(_pending, fixed=True, experiment_config=_experiment_config)
    elif args.op == "search":
        if _experiment_config['experiment_mode'] != 'hyperparameter_search':
            raise ValueError(f'The "experiment_mode" must be set to "hyperparameter_search"'
                             'in {args.config_file} when using the "--op search" option')
        generate_configs(_pending, fixed=False, experiment_config=_experiment_config)
    elif args.op == "status":
        report_pending_status(_pending)
    elif args.op == "reset":
        reset_running_status(_pending)
    elif args.op == "clear":
        clear_pending_configs(_pending)
    else:
        raise ValueError("Undefined operation!")
