"""Shared-queue bridge for the explicitly selected prediction Transformer."""
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
    from dual_branch.transformer_cli import main
    main()
