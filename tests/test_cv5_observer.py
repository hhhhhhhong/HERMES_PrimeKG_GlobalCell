"""Smoke-check files must never release the remaining formal CV folds."""
import copy,tempfile,unittest
from pathlib import Path
from formal_cv5 import report
from global_cell.common import ROOT,read,write

class FormalEntryGateTest(unittest.TestCase):
 def test_requires_two_formal_full_fit_updates(self):
  original=report.RUN
  with tempfile.TemporaryDirectory(dir=ROOT/'artifacts') as tmp:
   report.RUN=Path(tmp)
   try:
    d=report.RUN/'ONEIL/fold0';d.mkdir(parents=True)
    exp=copy.deepcopy(read(ROOT/'artifacts/cpu_check_ONEIL/EXPERIMENT.json'))
    write(d/'EXPERIMENT.json',exp)
    write(d/'STRUCTURE.json',read(ROOT/'artifacts/cpu_check_ONEIL/STRUCTURE.json'))
    for epoch in [1,2]:write(d/f'GRADIENT_epoch{epoch}.json',read(ROOT/f'artifacts/cpu_check_ONEIL/GRADIENT_epoch{epoch}.json'))
    write(d/'progress.json',dict(optimizer_updates=1));report.formal_entry('ONEIL')
    self.assertFalse((report.RUN/'ENTRY_ONEIL.json').exists())
    # The actual two-step check has the same nominal recipe, but cannot be
    # mistaken for formal fold0: its stage is check and only128 rows were used.
    write(d/'progress.json',dict(optimizer_updates=2))
    with self.assertRaises(AssertionError):report.formal_entry('ONEIL')
    self.assertFalse((report.RUN/'ENTRY_ONEIL.json').exists())
    exp['stage']='train';write(d/'EXPERIMENT.json',exp)
    grad=read(d/'GRADIENT_epoch2.json');bad=dict(grad,expression_edge=0.)
    write(d/'GRADIENT_epoch2.json',bad)
    with self.assertRaises(AssertionError):report.formal_entry('ONEIL')
    write(d/'GRADIENT_epoch2.json',grad);report.formal_entry('ONEIL')
    self.assertEqual(read(report.RUN/'ENTRY_ONEIL.json')['status'],'complete')
   finally:report.RUN=original

if __name__=='__main__':unittest.main()
