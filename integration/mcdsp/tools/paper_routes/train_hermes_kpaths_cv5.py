"""Bridge only: all GPU admission belongs to the shared paper-routes queue."""
import sys

if __name__=='__main__':
    sys.path.insert(0,'/ifs2/home/liushuhong/work2/HERMES_PrimeKG_GlobalCell')
    from dual_branch.kpaths_cli import main
    main()
