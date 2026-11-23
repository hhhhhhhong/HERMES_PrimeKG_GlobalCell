"""Only a bridge; the existing shared queue owns every GPU process."""
import sys

if __name__ == '__main__':
    sys.path.insert(0, '/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
    from dual_branch.repair_cli import main
    main()
