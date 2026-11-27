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
import torch.nn as nn
from tqdm import tqdm

from TrustworthyNet import TrustworthyNet_cluster
from util.clusteringPerformance import StatisticClustering
from util.loadMatData import (features_to_adj_legacy, getInitF,
                                  normalization, standardization_legacy)

INIT_NAMES = {
    'MNIST10k': 'MNIST10kWG',
    'NUS-WIDE': 'NUSWIDEWG',
    'Caltech101-all': 'Caltech101allWG',
    'MITIndoor': 'MITIndoorWG',
    'scene15': 'scene15WG',
    'small_Reuters': 'smallReutersWG',
    'esp-game': 'espgameWG',
    'flickr': 'flickrWG',
}


def train(model, features, adjacency, labels, n_clusters, args, device):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, betas=(0.90, 0.92),
        eps=0.01, weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.3, patience=15, min_lr=1e-8,
    )
    criterion = nn.MSELoss()
    best_acc = -1.0
    best_result, best_epoch = None, None
    loss_list = []
    fields = ['ACC', 'NMI', 'Purity', 'ARI', 'Fscore', 'Precision', 'Recall']
    start = time.time()

    with tqdm(total=args.epoch, desc='Training') as pbar:
        for epoch in range(args.epoch):
            model.train()
            optimizer.zero_grad()
            z = model(features, None, adjacency, args.active)
            output_z = z[-1]
            loss_self1 = torch.Tensor(np.array([0])).to(device)
            loss_self2 = torch.Tensor(np.array([0])).to(device)

            # The ORIGINAL protocol uses blocks-1, not blocks.
            # Preserve the loop's floating-point accumulation ordering.
            for k in range(len(features)):
                for j in range(args.block - 1):
                    loss_self1 += criterion(z[j].mm(z[j].t()), features[k].mm(features[k].t()))
                    loss_self2 += criterion(z[j].mm(z[j].t()), adjacency[k])
            loss_self1 = args.gamma * loss_self1
            loss_self2 = args.gamma2 * loss_self2
            loss = loss_self1 + loss_self2
            loss.requires_grad_(True)
            loss.backward()
            optimizer.step()
            scheduler.step(loss)
            first_loss = loss_self1.detach().cpu().numpy()
            second_loss = loss_self2.detach().cpu().numpy()
            total_loss = loss.detach().cpu().numpy()
            loss_list.append(float(total_loss[0]))

            # The original result is the mean of ten seeded KMeans runs at
            # the epoch with the highest mean ACC, not the last epoch.
            scores = StatisticClustering(output_z.detach().cpu().numpy(), labels, n_clusters)
            current = [str(score[0] * 100) + '(' + str(score[1] * 100) + ')' for score in scores]
            if scores[0][0] > best_acc:
                best_acc = float(scores[0][0])
                best_result = current
                best_epoch = epoch
            # print({'Self1 loss': '{:.6f}'.format(first_loss[0]),
            #        'Self2 loss': '{:.6f}'.format(second_loss[0]),
            #        'Loss': '{:.6f}'.format(total_loss[0])})
            # print({'Self1 loss': '{:.6f}'.format(first_loss[0]),
            #        'Self2 loss': '{:.6f}'.format(second_loss[0]),
            #        'Loss': '{:.6f}'.format(total_loss[0]),
            #        'ACC': '{:.2f} | {:.2f}'.format(scores[0][0] * 100, best_acc * 100)})
            # print(dict(zip(fields, current)))
            print({'Loss': '{:.6f}'.format(total_loss[0])})
            pbar.update(1)

    print('------------------------')
    print('best_score:', dict(zip(fields, best_result or [])))
    print('last_loss:', loss_list[-1] if loss_list else None)
    print('running_time:', time.time() - start)
    print('------------------------')
    return {'best_score': dict(zip(fields, best_result or [])),
            'best_epoch': best_epoch, 'loss_list': loss_list}


def run_dataset(args):
    if args.input_type != 'feature':
        raise ValueError('The restored experiment uses --input_type feature')
    if args.fusion_type not in ('trust', 'average', 'weight', 'attention'):
        raise ValueError('fusion_type must be trust, average, weight or attention')
    if args.active not in ('l1', 'l21'):
        raise ValueError('Clustering supports l1 or l21')
    if args.block < 2:
        raise ValueError('--block must be >=2 for the original block-1 loss')
    if args.fix_seed:
        # Preserve the historical seeding order before building the model.
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)

    device = torch.device('cpu' if args.device == 'cpu' else 'cuda:' + args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available. Use --device cpu for a small CPU smoke test.')
    data_dir = os.path.join(args.path, args.data_path.lstrip('/\\'))
    name = args.data_name
    init_name = args.init_name or INIT_NAMES.get(name, name + 'WG')

    print('loading {} data...'.format(name))
    adjacency_sparse, feature_arrays, labels = features_to_adj_legacy(name, data_dir, args.knns)  # normalized D^-1/2 A D^-1/2
    n = len(labels)
    dims = [feature.shape[1] for feature in feature_arrays]
    n_clusters = len(np.unique(labels))
    n_view = len(feature_arrays)
    print(n, dims, n_view)
    features = []
    adjacency = []
    for i, feature in enumerate(feature_arrays):
        # Kept for exact historical reproduction; '//': floor division,
        # not conventional min-max scaling. Do NOT silently change to '/'.
        x = torch.from_numpy(np.asarray(feature / 1.0)).float().to(device)
        x = standardization_legacy(normalization(x))
        features.append(x)
        adjacency.append(adjacency_sparse[i].to_dense().float().to(device))

    z_init = getInitF(init_name, n_view, data_dir)
    model = TrustworthyNet_cluster(dims, n_view, n_clusters, n, z_init, args, device).to(device)
    print('*' * 40)
    print('fusion:{},active:{},gamma1:{},gamma2:{},block:{},epoch:{},thre:{},lr:{}'.format(
        args.fusion_type, args.active, args.gamma, args.gamma2, args.block, args.epoch, args.thre, args.lr))
    return train(model, features, adjacency, labels, n_clusters, args, device)





# ========== One-click settings: EDIT HERE; NO command-line arguments ==========
RUN_ALL = True # True: run every dataset listed in ALL_DATASETS
SINGLE_DATASET = 'NUS-WIDE'    # Used only when RUN_ALL = False
ALL_DATASETS = (
    'Caltech101-all',
    'MITIndoor',
    'MNIST10k',
    'NUS-WIDE',
    'scene15',
    'small_Reuters',
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
    print('Running clustering datasets:', ', '.join(selected), flush=True)

    for index, dataset in enumerate(selected, 1):
        print('\n' + '=' * 72, flush=True)
        print('[{}/{}] clustering: {}'.format(index, len(selected), dataset), flush=True)
        try:
            args = load_dataset_args('clustering', dataset)
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

    print('\nFinished clustering: {}/{} succeeded'.format(successes, len(selected)), flush=True)


if __name__ == '__main__':
    main()
