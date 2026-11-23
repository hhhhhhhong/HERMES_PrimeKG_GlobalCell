"""One graph forward and one optimizer step per epoch, with chunked readout."""
import torch
from .stable_training import train_epoch as shared_train_epoch


def train_epoch(model, optimizer, rows, labels, chunk=1024):
    before = model.table_builds
    counters = {n: int(b.num_batches_tracked) for n,b in model.collaborative_refiner.named_modules()
                if isinstance(b, torch.nn.BatchNorm1d)}
    loss, diagnostic = shared_train_epoch(model, optimizer, rows, labels, chunk=chunk)
    assert model.table_builds-before == 1
    for name, b in model.collaborative_refiner.named_modules():
        if name in counters:
            assert int(b.num_batches_tracked)-counters[name] == 1
    diagnostic.update(graph_tables_recomputed=1, Refiner_BN_updates_per_epoch=1,
                      target_exclusion=False, query_batches=(len(rows)+chunk-1)//chunk)
    return loss, diagnostic


@torch.no_grad()
def evaluate(model, rows, chunk=1024):
    model.eval()
    base = model.base_features()
    table = model.knowledge_table(base)
    return torch.cat([model(rows[i:i+chunk], table, base)[0].cpu() for i in range(0,len(rows),chunk)])


class AucEarlyStop:
    """Patience counts epochs since best AUROC, not number of evaluations."""
    def __init__(self, patience=100, minimum_epochs=300):
        self.patience, self.minimum_epochs = patience, minimum_epochs
        self.best, self.best_epoch = -float('inf'), 0

    def observe(self, epoch, auc):
        import math
        if not math.isfinite(auc):
            raise ValueError('Nonfinite validation AUROC')
        improved = auc > self.best
        if improved:
            self.best, self.best_epoch = auc, epoch
        stop = epoch >= self.minimum_epochs and epoch-self.best_epoch >= self.patience
        return improved, stop
