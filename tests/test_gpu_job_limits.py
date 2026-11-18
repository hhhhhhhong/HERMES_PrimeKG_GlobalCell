"""A two-job exception must not widen every queued experiment's resource cap."""
import importlib.util,sys,unittest
from pathlib import Path
sys.path.insert(0,'/ifs2/home/liushuhong/work2/mcdsp_moe/MCDSP-main')
from tools.paper_routes.queue import job_gpu_slot_available as available
from global_cell.common import ROOT

class JobProcessLimitTests(unittest.TestCase):
 def test_default_and_explicit_approved_slots(self):
  occupied={1:{10,11}}
  self.assertFalse(available({},1,occupied,{},{}))
  self.assertTrue(available({},1,occupied,{},{1:4}))
  self.assertFalse(available({},1,{1:{10,11,12,13}},{},{1:4}))
 def test_other_jobs_keep_original_ceiling(self):
  job={'gpu_process_limits':{'1':2}}
  self.assertFalse(available(job,1,{1:{10,11}},{},{1:4}))
  self.assertTrue(available(job,1,{1:{10}},{},{1:4}))
  self.assertFalse(available({'gpu_process_limits':{'1':8}},1,{1:{10,11}},{},{}))
 def test_inflight_processes_count_and_other_gpu_unchanged(self):
  self.assertFalse(available({},1,{1:{10,11,12}},{'starting':{'gpu':1,'pid':13}},{1:4}))
  self.assertTrue(available({},3,{3:{1,2,3,4,5}},{},{1:4,3:6}))
 def test_invalid_ceiling_rejected(self):
  for value in [0,-1,True,'2']:
   with self.assertRaises(ValueError):available({'gpu_process_limits':{'1':value}},1,{},{},{1:4})
 def test_independent_scope_expiry(self):
  p=ROOT/'scripts/expire_concurrency_scopes.py';spec=importlib.util.spec_from_file_location('scope_expiry_test',p)
  module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
  cmd=['python','-m','queue','--gpu-process-limit','3=6','--gpu-process-limit','1=4']
  self.assertEqual(module.remove_limits(cmd,{'1'}),cmd[:-2])
  self.assertEqual(module.remove_limits(cmd,{'3'}),['python','-m','queue','--gpu-process-limit','1=4'])
  self.assertEqual(module.remove_limits(cmd,{'1','3'}),['python','-m','queue'])
if __name__=='__main__':unittest.main()
