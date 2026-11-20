"""Shared queue bridge for the CODER / molecular-expression dual branch."""
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
    from dual_branch.cli import main
    main()
