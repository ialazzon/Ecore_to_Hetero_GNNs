"""
Generic MULTI-CLASS graph classification over a folder of PyG HeteroData
`.pt` snapshots (e.g. xmi_new/hd_0000.pt, hd_0001.pt, ...).

Schema-agnostic: node types, edge types, feature dims, and edge_attr dims are
all introspected from the data at load time -- nothing about Room/Sensor/
Connection is hardcoded.

Labels come from LABEL_CSV, e.g.:
    graph_name,label
    snapshot_0000,HVACFault
    snapshot_0001,Normal
    ...
Each row's graph_name is matched to a file `hd_<same suffix>.pt` in DATA_DIR
(this mirrors your hd_NNNN.pt / snapshot_NNNN naming convention). The set of
class names is discovered automatically from the CSV -- add a class, it just
works, no code change.

Pipeline:
  1. Load every label_of.csv row, match it to its hd_*.pt file, load the graph.
  2. AUTO-MIRROR single-directed edge types (e.g. 'has') into bidirectional
     ones ('rev_has'), by flipping edge_index and copying edge_attr if
     present. Relations that already have an explicit reverse (like your
     'rev_Connection') are left untouched -- never double-mirrored.
  3. Stratified 70/15/15 split over graphs (by class). The exact graphs that
     land in each split are SAVED as copies (pre-normalization, so they're
     reusable as-is) into:
         exp_graphs/Training/hd_*.pt
         exp_graphs/Test/hd_*.pt
         exp_graphs/Validation/hd_*.pt
     and the snapshot names/numbers in each split are printed.
     Train-only feature normalization (node x and edge_attr) is then fit on
     the training split and applied to val/test (in memory, for training --
     the saved copies on disk are NOT normalized, so they stay reusable).
  4. ONE hidden HeteroConv message-passing layer -- GATConv per relation,
     edge_dim set for any relation with edge_attr, None otherwise -- then
     mean+max pooling per node type, then a linear head over num_classes.
  5. Train, keep best-val checkpoint, report test accuracy + per-class
     precision/recall/F1.

To point this at a different relational dataset: change DATA_DIR, LABEL_CSV,
EXP_DIR, and the filename-matching logic in `load_dataset` if your naming
differs. Nothing else needs touching.
"""

import os
import glob
import csv
import copy
import random
import re
from collections import defaultdict, Counter

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GATConv, HeteroConv
from torch_geometric.nn import global_mean_pool, global_max_pool

# --------------------------------------------------------------------------- #
# 0. Config
# --------------------------------------------------------------------------- #

DATA_DIR = 'pt_graphs'          # folder containing hd_*.pt HeteroData files
LABEL_CSV = 'label_of.csv'    # graph_name,label  (e.g. snapshot_0000,HVACFault)

EXP_DIR = 'exp_graphs'          # where split copies get saved
EXP_TRAIN_DIR = os.path.join(EXP_DIR, 'Training')
EXP_TEST_DIR = os.path.join(EXP_DIR, 'Test')
EXP_VAL_DIR = os.path.join(EXP_DIR, 'Validation')

SEED = 0
DIM_H = 64
DROPOUT = 0.5
EPOCHS = 201
BATCH_SIZE = 16

random.seed(SEED)
torch.manual_seed(SEED)


# --------------------------------------------------------------------------- #
# 1. Load labels, match to graphs, discover classes automatically
# --------------------------------------------------------------------------- #

def load_dataset(data_dir, label_csv):
    label_of = {}
    with open(label_csv, newline='') as f:
        reader = csv.reader(f)
        rows = [row for row in reader if row]
    # Only treat the first row as a header if it doesn't look like real data
    # (i.e. its first column has no trailing digits, like "graph_name" rather
    # than "snapshot_0000"). This avoids silently eating a real labeled row
    # when the CSV has no header line -- exactly the bug that dropped
    # snapshot_0000 before.
    start = 0
    if rows and not re.search(r'\d+$', rows[0][0].strip()):
        start = 1
    for row in rows[start:]:
        if len(row) >= 2:
            label_of[row[0].strip()] = row[1].strip()

    # Match graphs to labels by the NUMERIC id embedded in the name, not by
    # reconstructing a padded string. This avoids bugs from mismatched
    # zero-padding between the .pt filenames (hd_0000.pt -> 4 digits) and
    # whatever padding label_of.csv happens to use (snapshot_0000,
    # snapshot_00000, snapshot_1, etc. all work identically). If a CSV key
    # has no trailing digits it's skipped with a warning -- fix the CSV row.
    label_by_num = {}
    for k, v in label_of.items():
        m = re.search(r'(\d+)$', k)
        if m:
            label_by_num[int(m.group(1))] = v
        else:
            print(f'  [warn] label_of.csv row "{k}" has no trailing number, skipping it')

    classes = sorted(set(label_by_num.values()))
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f'Discovered {len(classes)} classes: {classes}')
    print(f'{len(label_by_num)} labeled rows parsed from {label_csv}')

    all_files = sorted(glob.glob(os.path.join(data_dir, 'hd_*.pt')))
    print(f'{len(all_files)} hd_*.pt files found in {data_dir}')

    graphs = []
    skipped = []
    for path in all_files:
        suffix = os.path.splitext(os.path.basename(path))[0][len('hd_'):]  # e.g. '0000'
        m = re.search(r'(\d+)$', suffix)
        num = int(m.group(1)) if m else None
        if num is None or num not in label_by_num:
            skipped.append(os.path.basename(path))
            continue
        g = torch.load(path, weights_only=False)
        g.y = torch.tensor([class_to_idx[label_by_num[num]]], dtype=torch.long)
        g.name = f'snapshot_{suffix}'
        graphs.append(g)

    print(f'Loaded {len(graphs)} labeled graphs ({len(skipped)} skipped, no label match).')
    if skipped:
        print(f'  first few skipped: {skipped[:5]}')
    if not graphs:
        raise RuntimeError(
            f'No graphs matched between {data_dir}/hd_*.pt and {label_csv}. '
            f'Check that both sides share the same numeric id, and that '
            f'label_of.csv columns are [graph_name, label] in that order.'
        )
    return graphs, classes


data, CLASSES = load_dataset(DATA_DIR, LABEL_CSV)


# --------------------------------------------------------------------------- #
# 2. Auto-mirror single-directed edge types -> bidirectional message passing
# --------------------------------------------------------------------------- #

def mirror_single_directed_edges(graphs):
    """Add a reverse edge type for any relation that doesn't already have one
    (neither a same-named symmetric counterpart nor an explicit 'rev_' one),
    copying edge_attr along if present. Already-bidirectional relations
    (e.g. your Connection/rev_Connection) are left untouched."""
    for g in graphs:
        existing = set(g.edge_types)
        to_add = []
        for et in list(existing):
            src, rel, dst = et
            has_reverse = (dst, rel, src) in existing or (dst, f'rev_{rel}', src) in existing
            if not has_reverse:
                to_add.append(et)
        for (src, rel, dst) in to_add:
            store = g[src, rel, dst]
            rev_et = (dst, f'rev_{rel}', src)
            g[rev_et].edge_index = store.edge_index.flip(0)
            if 'edge_attr' in store and store.edge_attr is not None and store.edge_attr.numel():
                g[rev_et].edge_attr = store.edge_attr.clone()
    return graphs


data = mirror_single_directed_edges(data)
metadata = data[0].metadata()
print('Node types:', metadata[0])
print('Edge types (after mirroring):', metadata[1])

# Infer edge_attr dim per relation (None if that relation carries no attrs).
# Assumes a consistent schema across all graphs in DATA_DIR.
EDGE_DIM = {}
for et in metadata[1]:
    ea = data[0][et].get('edge_attr', None)
    EDGE_DIM[et] = ea.size(-1) if (ea is not None and ea.numel()) else None
print('Edge feature dims per relation:', EDGE_DIM)


# --------------------------------------------------------------------------- #
# 3. Stratified split, save split copies to disk, train-only normalization
# --------------------------------------------------------------------------- #

by_class = defaultdict(list)
for i, g in enumerate(data):
    by_class[int(g.y)].append(i)

train_idx, val_idx, test_idx = [], [], []
for cls, idxs in by_class.items():
    random.shuffle(idxs)
    n = len(idxs)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    train_idx += idxs[:n_train]
    val_idx += idxs[n_train:n_train + n_val]
    test_idx += idxs[n_train + n_val:]

random.shuffle(train_idx); random.shuffle(val_idx); random.shuffle(test_idx)
train_data = [data[i] for i in train_idx]
val_data = [data[i] for i in val_idx]
test_data = [data[i] for i in test_idx]
print(f'train: {len(train_data)}  val: {len(val_data)}  test: {len(test_data)}')
print('class balance (overall):', {CLASSES[k]: v for k, v in
      sorted((k, len(v)) for k, v in by_class.items())})


def save_split(graphs, out_dir, split_name):
    """Save a copy of each graph in this split to out_dir as hd_<suffix>.pt
    (mirroring the original hd_NNNN.pt naming), and print/return the sorted
    list of snapshot names that ended up in this split."""
    os.makedirs(out_dir, exist_ok=True)
    names = []
    for g in graphs:
        # g.name is 'snapshot_<suffix>' -> file is 'hd_<suffix>.pt'
        suffix = g.name[len('snapshot_'):]
        out_path = os.path.join(out_dir, f'hd_{suffix}.pt')
        torch.save(copy.deepcopy(g), out_path)
        names.append(g.name)
    names = sorted(names)
    print(f'\n[{split_name}] saved {len(names)} graphs to {out_dir}')
    print(f'[{split_name}] snapshots: {names}')
    return names


train_names = save_split(train_data, EXP_TRAIN_DIR, 'train')
val_names = save_split(val_data, EXP_VAL_DIR, 'val')
test_names = save_split(test_data, EXP_TEST_DIR, 'test')


def _fit_stats(graphs):
    node_feats, edge_feats = defaultdict(list), defaultdict(list)
    for g in graphs:
        for nt in g.node_types:
            if 'x' in g[nt] and g[nt].x.numel():
                node_feats[nt].append(g[nt].x)
        for et in g.edge_types:
            if 'edge_attr' in g[et] and g[et].edge_attr is not None and g[et].edge_attr.numel():
                edge_feats[et].append(g[et].edge_attr)
    stats = {'node': {}, 'edge': {}}
    for nt, chunks in node_feats.items():
        allx = torch.cat(chunks, dim=0)
        stats['node'][nt] = (allx.mean(0, keepdim=True), allx.std(0, keepdim=True).clamp_min(1e-6))
    for et, chunks in edge_feats.items():
        alle = torch.cat(chunks, dim=0)
        stats['edge'][et] = (alle.mean(0, keepdim=True), alle.std(0, keepdim=True).clamp_min(1e-6))
    return stats


def _apply_stats(graphs, stats):
    for g in graphs:
        for nt in g.node_types:
            if nt in stats['node'] and 'x' in g[nt] and g[nt].x.numel():
                mean, std = stats['node'][nt]
                g[nt].x = (g[nt].x - mean) / std
        for et in g.edge_types:
            if et in stats['edge'] and 'edge_attr' in g[et] and g[et].edge_attr is not None and g[et].edge_attr.numel():
                mean, std = stats['edge'][et]
                g[et].edge_attr = (g[et].edge_attr - mean) / std


stats = _fit_stats(train_data)
_apply_stats(train_data, stats)
_apply_stats(val_data, stats)
_apply_stats(test_data, stats)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)


# --------------------------------------------------------------------------- #
# 4. Model: ONE hidden HeteroConv message-passing layer, mean+max pool,
#    linear head over num_classes
# --------------------------------------------------------------------------- #

class HeteroGraphClassifier(nn.Module):
    def __init__(self, metadata, edge_dim_map, num_classes, dim_h=64, dropout=0.5):
        super().__init__()
        node_types, edge_types = metadata
        conv_dict = {
            et: GATConv((-1, -1), dim_h, add_self_loops=False,
                        edge_dim=edge_dim_map.get(et), dropout=dropout)
            for et in edge_types
        }
        self.conv = HeteroConv(conv_dict, aggr='sum')   # single hidden layer
        self.dropout = dropout
        self.linear = nn.Linear(dim_h * 2, num_classes)

    def forward(self, x_dict, edge_index_dict, edge_attr_dict, batch_dict):
        h_dict = self.conv(x_dict, edge_index_dict, edge_attr_dict=edge_attr_dict)
        h_dict = {k: F.relu(v) for k, v in h_dict.items()}
        h_dict = {k: F.dropout(v, p=self.dropout, training=self.training)
                  for k, v in h_dict.items()}
        mean_parts, max_parts = [], []
        for nt, h in h_dict.items():
            mean_parts.append(global_mean_pool(h, batch_dict[nt]))
            max_parts.append(global_max_pool(h, batch_dict[nt]))
        h_graph = torch.cat([torch.stack(mean_parts, dim=0).sum(dim=0),
                              torch.stack(max_parts, dim=0).sum(dim=0)], dim=-1)
        h_graph = F.dropout(h_graph, p=self.dropout, training=self.training)
        return self.linear(h_graph)   # [batch, num_classes] logits


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def _edge_attr_dict(batch):
    return {et: batch[et].edge_attr for et in batch.edge_types
            if 'edge_attr' in batch[et] and batch[et].edge_attr is not None
            and batch[et].edge_attr.numel()}


@torch.no_grad()
def evaluate(model, loader, num_classes):
    model.eval()
    tp = torch.zeros(num_classes); fp = torch.zeros(num_classes)
    fn = torch.zeros(num_classes); support = torch.zeros(num_classes)
    correct = total = 0
    for batch in loader:
        batch = batch.to(device)
        logits = model(batch.x_dict, batch.edge_index_dict,
                        _edge_attr_dict(batch), batch.batch_dict)
        pred = logits.argmax(dim=-1)
        y = batch.y.view(-1)
        correct += int((pred == y).sum())
        total += y.numel()
        for c in range(num_classes):
            tp[c] += int(((pred == c) & (y == c)).sum())
            fp[c] += int(((pred == c) & (y != c)).sum())
            fn[c] += int(((pred != c) & (y == c)).sum())
            support[c] += int((y == c).sum())
    acc = correct / total if total else 0.0
    precision = tp / (tp + fp).clamp_min(1)
    recall = tp / (tp + fn).clamp_min(1)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-9)
    return acc, precision, recall, f1, support


num_classes = len(CLASSES)
model = HeteroGraphClassifier(metadata, EDGE_DIM, num_classes, dim_h=DIM_H,
                               dropout=DROPOUT).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=5e-3)

best_val_acc = 0.0
best_state = copy.deepcopy(model.state_dict())

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0.0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        logits = model(batch.x_dict, batch.edge_index_dict,
                        _edge_attr_dict(batch), batch.batch_dict)
        loss = F.cross_entropy(logits, batch.y.view(-1))
        loss.backward()
        optimizer.step()
        total_loss += float(loss.detach()) * batch.num_graphs
    total_loss /= len(train_data)

    val_acc, val_p, val_r, val_f1, _ = evaluate(model, val_loader, num_classes)
    if val_acc >= best_val_acc:
        best_val_acc = val_acc
        best_state = copy.deepcopy(model.state_dict())

    if epoch % 20 == 0:
        train_acc, _, _, _, _ = evaluate(model, train_loader, num_classes)
        print(f'Epoch {epoch:>3} | Loss {total_loss:.4f} | '
              f'Train Acc {train_acc*100:.2f}% | Val Acc {val_acc*100:.2f}%')

model.load_state_dict(best_state)
test_acc, test_p, test_r, test_f1, support = evaluate(model, test_loader, num_classes)
macro_f1 = test_f1.mean().item()

print('\n================ RESULT ================')
print(f'Best Val Acc: {best_val_acc*100:.2f}%')
print(f'Test Acc: {test_acc*100:.2f}%\n')
print(f'Macro-F1: {macro_f1:.4f}\n')
print(f'{"class":<16}{"precision":>10}{"recall":>10}{"f1":>10}{"support":>10}')
for c, cname in enumerate(CLASSES):
    print(f'{cname:<16}{test_p[c]:>10.3f}{test_r[c]:>10.3f}{test_f1[c]:>10.3f}{int(support[c]):>10}')

print('\n================ SPLIT SNAPSHOTS ================')
print(f'Training  ({len(train_names)}): {train_names}')
print(f'Test      ({len(test_names)}): {test_names}')
print(f'Validation({len(val_names)}): {val_names}')
