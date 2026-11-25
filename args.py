import argparse
import os
import yaml


## Parameter setting
def parameter_parser():
    parser = argparse.ArgumentParser()

    current_dir = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--path", type=str, default=current_dir,
                        help="Project root used to locate the data directory.")
    parser.add_argument("--data_path", type=str, default="/data/",
                        help="Dataset directory relative to --path.")
    parser.add_argument("--input_type", type=str, default="feature",
                        help="feature or similarity. Both tasks use feature.")
    parser.add_argument("--fusion_type", type=str, default="trust",
                        help="View fusion: trust, average, weight, or attention.")
    parser.add_argument("--active", type=str, default="l1",
                        help="Activation for both tasks: l1 or l21.")
    parser.add_argument("--fix_seed", action="store_true", default=True,
                        help="Fix the random seed before model initialization.")
    parser.add_argument("--seed", type=int, default=40,
                        help="Random seed used when --fix_seed is set.")
    parser.add_argument("--ratio", type=float, default=0.1,
                        help="Fraction of labeled samples per class for classification.")
    parser.add_argument("--weight_decay", type=float, default=0.15,
                        help="Adam weight decay.")
    parser.add_argument("--epoch", type=int, default=100,
                        help="Number of training epochs.")
    parser.add_argument("--gamma", type=float, default=1,
                        help="Weight of the supervised loss, or the clustering self-expression loss.")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Initial learning rate.")
    parser.add_argument("--thre", type=float, default=0.1,
                        help="Initial activation threshold (theta).")
    parser.add_argument("--init_name", type=str, default=None,
                        help="Clustering initialization MAT filename without .mat. Defaults to the dataset WG file.")
    parser.add_argument("--knns", type=int, default=7,
                        help="Neighbors for the clustering binary k-NN graph, normalized after symmetrization.")
    parser.add_argument("--gamma2", type=float, default=1,
                        help="Weight of the adjacency reconstruction term in clustering.")
    # Entry scripts configure datasets in Python, not via CLI arguments.
    # Ignore shell arguments, including IDE-injected arguments.
    return parser.parse_args(args=[])


def load_dataset_args(task, dataset):
    """Merge common args.py defaults with per-dataset YAML (block, device ONLY)."""
    if task not in ('clustering', 'semiClassifier'):
        raise ValueError('Unsupported task: {}'.format(task))
    args = parameter_parser()
    config_dir = 'config_c' if task == 'clustering' else 'config_s'
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), config_dir, dataset + '.yaml')
    if os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
    else:
        if task == 'semiClassifier':
            config = {
                'block': 2,
                'device': '0'
            }
            print(f"[INFO] No YAML found for {dataset}, "
                  f"using default block=2, device=0")
        else:
            raise FileNotFoundError(f"Missing configuration: {path}")

    for key, value in config.items():
        if key not in ('block', 'device') and not hasattr(args, key):
            raise ValueError(
                f"Unknown parameter '{key}' in {path}"
            )

        setattr(args, key, value)
    if type(config['block']) is not int or config['block'] < 1:
        raise ValueError('{}: block must be a positive integer'.format(path))
    device = str(config['device'])
    if device != 'cpu' and not device.isdigit():
        raise ValueError('{}: device must be cpu or a CUDA index'.format(path))
    args.data_name = dataset
    args.block = config['block']
    args.device = device

    return args
