import argparse
import gnnbench
import json
import os
import shlex

from gnnbench.util import get_pending_collection, get_experiment_config
from pathlib import Path

# Get path to the `run_single_job.py` file which is called internally to execute the jobs.
SCRIPT_PATH = Path(gnnbench.__file__).parent / 'run_single_job.py'


def do_work(db_host, db_port, device, log_verbose):
    pending = get_pending_collection(db_host, db_port)

    if pending.count() <= 0:
        print("No pending jobs. Exiting...")
        return

    if device is None:
        raise ValueError("GPU ID not set!")

    print(f"Running configs from database on GPU {device}.\n\n")
    working_loop(pending, db_host, db_port, device=device, log_verbose=log_verbose)


def working_loop(pending, db_host, db_port, device, log_verbose):
    job = pending.find_one_and_update({"running": False}, {"$set": {"running": True}})
    while job is not None:
        config = job["config"]
        config['db_host'] = db_host
        config['db_port'] = db_port

        print(f"{config['experiment_name']}: Running split {config['split_no']} with seed {config['seed']}...\n---")
        config_string = f"{json.dumps(config)}"
        # escape bash metacharacters (mostly "")
        config_string = shlex.quote(config_string)

        command = f"python {SCRIPT_PATH} {device} {1 if log_verbose else 0} {config_string}"
        print(command)
        os.system(command)
        print(f"{config['experiment_name']}: Done split {config['split_no']} with seed {config['seed']}...")
        pending.delete_one({"_id": job["_id"]})

        # look if there's still work
        job = pending.find_one_and_update({"running": False}, {"$set": {"running": True}})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Create a worker process that fetches pending jobs from the '
                                                 '"pending" database and executes them one by one. '
                                                 'Each worker runs on a single specified GPU device.')
    parser.add_argument('-c',
                        '--config-file',
                        type=str,
                        required=True,
                        help='Path to the YAML configuration file for the experiment.')
    parser.add_argument('-g',
                        '--gpu',
                        type=int,
                        required=True,
                        help='The ID of the GPU to operate on.')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Display more log messages.')
    args = parser.parse_args()

    _experiment_config = get_experiment_config(args.config_file)
    _db_host = _experiment_config['db_host']
    _db_port = _experiment_config['db_port']
    do_work(_db_host, _db_port, args.gpu, args.verbose)
