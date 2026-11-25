import os

os.environ["MPLCONFIGDIR"] = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    ".matplotlib"
)

os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from TrustworthyNet import TrustworthyNet_classfier
from util.loadMatData import load_data_2, features_to_Lap, generate_partition
from util.utils import get_evaluation_results


def train(model, feature_list, sim_list, lap_list, labels, idx_train, idx_test, args, device):
    criterion = torch.nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    labels = torch.as_tensor(labels, dtype=torch.long, device=device)
    best_acc = -1.0
    best_f1_macro = 0.0
    best_f1_micro = 0.0
    best_epoch = None
    loss_list = []
    start_time = time.time()

    with tqdm(total=args.epoch, desc='Training') as pbar:
        for epoch in range(args.epoch):
            model.train()
            Z = model(feature_list, sim_list, lap_list, args.active)
            output = F.log_softmax(Z[-1], dim=1)
            loss_sup = criterion(output[idx_train], labels[idx_train])
            loss = args.gamma * loss_sup

            optimizer.zero_grad()
            loss.requires_grad_(True)  # historical training statement
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())

            with torch.no_grad():
                model.eval()
                Z = model(feature_list, sim_list, lap_list, args.active)
                pred_labels = torch.argmax(Z[-1], dim=1).cpu().numpy()
                ACC, F1_macro, F1_micro = get_evaluation_results(
                    labels[idx_test].cpu().numpy(), pred_labels[idx_test]
                )
                if ACC >= best_acc:
                    best_acc = ACC
                    best_f1_macro = F1_macro
                    best_f1_micro = F1_micro
                    best_epoch = epoch
                print({'Loss': '{:.6f}'.format(loss.item())})
                # print({'Loss': '{:.6f}'.format(loss.item()),
                #        'ACC': '{:.2f} | {:.2f}'.format(ACC * 100, best_acc * 100),
                #        'F1_macro': '{:.2f} | {:.2f}'.format(F1_macro * 100, best_f1_macro * 100),
                #        'F1_micro': '{:.2f} | {:.2f}'.format(F1_micro * 100, best_f1_micro * 100)})
            pbar.update(1)

    print('------------------------')
    print('Best ACC: {:.2f}'.format(best_acc * 100))
    print('Best F1_macro (at best ACC): {:.2f}'.format(best_f1_macro * 100))
    print('Best F1_micro (at best ACC): {:.2f}'.format(best_f1_micro * 100))
    print('running_time: {:.2f}'.format(time.time() - start_time))
    print('------------------------')
    return {'ACC': best_acc, 'F1_macro': best_f1_macro,
            'F1_micro': best_f1_micro, 'best_epoch': best_epoch,
            'loss_list': loss_list}


def run_dataset(args):
    if args.active not in ('l1', 'l21'):
        raise ValueError('Classification supports l1 or l21')
    if args.input_type != 'feature':
        raise ValueError('The historical semi-supervised protocol uses --input_type feature')
    if args.block < 1:
        raise ValueError('--block must be positive')
    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA is not available; use --device cpu')
    data_dir = os.path.join(args.path, args.data_path.lstrip('/'))
    print('loading {} data...'.format(args.data_name))
    raw_views, labels = load_data_2(args.data_name, data_dir + os.sep)
    n_view = len(raw_views)
    n_feats = [x.shape[1] for x in raw_views]
    n = raw_views[0].shape[0]
    n_classes = len(np.unique(labels))
    print(n, n_feats, n_view, 'classes:', n_classes)

    # Old semiClassifier.py: int(n / n_classes * 0.1), after row-L2 normalization.
    knns = int(n / n_classes * 0.1)
    if not (1 <= knns < n):
        raise ValueError(f'Invalid KNN count {knns} for {n} samples')
    print('constructing distance-weighted KNN graph with k={}'.format(knns))
    sim_sparse, lap_sparse = features_to_Lap(raw_views, knns)
    features = [torch.as_tensor(np.asarray(x), dtype=torch.float32, device=device)
                for x in raw_views]
    sims = [x.to_dense().to(device) for x in sim_sparse]
    laps = [x.to_dense().to(device) for x in lap_sparse]

    # Match old training: seed after graph construction, before model init.
    if args.fix_seed:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
    idx_train, idx_test = generate_partition(labels=labels, ratio=args.ratio)
    if not idx_train or not idx_test:
        raise ValueError('Empty labeled or unlabeled split; change --ratio')
    print('train (labeled): {}, eval (unlabeled): {}'.format(len(idx_train), len(idx_test)))

    model = TrustworthyNet_classfier(n_feats, n_view, n_classes, n, args, device).to(device)
    print('*' * 40)
    print('fusion:{},active:{},gamma:{},block:{},epoch:{},thre:{},lr:{},ratio:{}'.format(
        args.fusion_type, args.active, args.gamma, args.block, args.epoch,
        args.thre, args.lr, args.ratio))
    return train(model, features, sims, laps, labels, idx_train, idx_test, args, device)





# ========== One-click settings: EDIT HERE; NO command-line arguments ==========
RUN_ALL = True # True: run every dataset listed in ALL_DATASETS
SINGLE_DATASET = 'ORL'    # Used only when RUN_ALL = False
ALL_DATASETS = (
    'animals',
    'Caltech101-all',
    'MITIndoor',
    'NUS-WIDE',
    'ORL',
    'scene15',
    'esp-game',
    'flickr',
)
# ============================================================================


def main():
    selected = list(ALL_DATASETS) if RUN_ALL else [SINGLE_DATASET]
    if not selected or len(selected) != len(set(selected)):
        raise ValueError('Dataset list is empty or contains duplicates')
    unknown = [name for name in selected if name not in ALL_DATASETS]
    if unknown:
        raise ValueError('Unknown datasets: {}'.format(unknown))

    import gc
    from args import load_dataset_args

    successes = 0
    print('Running semiClassifier datasets:', ', '.join(selected), flush=True)

    for index, dataset in enumerate(selected, 1):
        print('\n' + '=' * 72, flush=True)
        print('[{}/{}] semiClassifier: {}'.format(index, len(selected), dataset), flush=True)
        try:
            args = load_dataset_args('semiClassifier', dataset)
            print('config:', {'block': args.block, 'device': args.device,
                              'epoch': args.epoch, 'lr': args.lr}, flush=True)
            run_dataset(args)
            successes += 1
        except Exception:
            import traceback
            print('FAILED: {}\n{}'.format(dataset, traceback.format_exc()), flush=True)
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    print('\nFinished semiClassifier: {}/{} succeeded'.format(successes, len(selected)), flush=True)


if __name__ == '__main__':
    main()
