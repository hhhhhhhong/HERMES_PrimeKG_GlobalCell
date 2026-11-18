"""Shared-queue bridge: unchanged zero-gated architecture, both datasets CV5."""
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
    from scripts.train_zero_gated_hgnn_cv5 import main
    main()
