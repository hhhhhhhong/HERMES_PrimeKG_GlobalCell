"""Shared queue bridge for the single ALMANAC three-round gated pilot."""
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
    from scripts.train_zero_gated_hgnn import main
    main()
