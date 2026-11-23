"""Bridge only; the shared paper-routes queue owns GPU admission."""
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
    from dual_branch.gated_cli import main
    main()
