"""ALMANAC fold0 repair pilot; all GPU admission is owned by shared queue."""
import sys
if __name__ == '__main__':
    sys.path.insert(0, '/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
    from dual_branch.backbone_repair_cli import main
    main()
