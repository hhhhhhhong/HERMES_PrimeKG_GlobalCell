"""Reuse verified frozen CODER inputs and encode only additional official names.

Runs on CPU so it does not borrow the old PathBatch GPU concurrency exception.
"""
import sys,time,os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
import torch
from global_cell.common import read,write,sha

def main():
 from transformers import AutoModel,BertTokenizerFast
 torch.set_num_threads(4);torch.manual_seed(0);start=time.monotonic()
 texts=read(ROOT/'data/ENTITY_TEXTS.json');missing=read(ROOT/'data/MISSING_CODER_TEXTS.json')
 model_dir=ROOT/'models/coder_eng';manifest=read(ROOT/'artifacts/CODER_DOWNLOAD.json')
 assert all(sha(model_dir/name)==h for name,h in manifest['files'].items())
 reused=torch.load(ROOT/'data/REUSED_CODER.pt',weights_only=True,map_location='cpu');vectors=dict(reused);truncated=[]
 if missing:
  tokenizer=BertTokenizerFast.from_pretrained(model_dir,do_lower_case=False,local_files_only=True)
  model=AutoModel.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False).eval();model.requires_grad_(False)
  with torch.inference_mode():
   for start_i in range(0,len(missing),32):
    group=missing[start_i:start_i+32];names=[g['text'] for g in group]
    lengths=[len(tokenizer.encode(n,add_special_tokens=True,truncation=False)) for n in names]
    truncated.extend(dict(prime_index=g['prime_index'],tokens=n) for g,n in zip(group,lengths) if n>64)
    tokens=tokenizer(names,padding=True,truncation=True,max_length=64,return_tensors='pt')
    result=torch.nn.functional.normalize(model(**tokens).pooler_output,p=2,dim=-1)
    assert torch.isfinite(result).all()
    vectors.update({g['prime_index']:v.clone() for g,v in zip(group,result)})
    write(ROOT/'artifacts/encoding_progress.json',dict(status='running',new_encoded=min(start_i+32,len(missing)),total_new=len(missing),seconds=time.monotonic()-start,pid=os.getpid(),device='cpu'))
 ids=[t['prime_index'] for t in texts];features=torch.stack([vectors[i] for i in ids]);assert features.shape==(len(ids),768)
 assert all(torch.equal(vectors[i],v) for i,v in reused.items())
 torch.save(dict(prime_indices=ids,features=features),ROOT/'data/CODER_ENTITY_FEATURES.pt');lookup={v:i for i,v in enumerate(ids)}
 for ds in ['ALMANAC','ONEIL']:
  graph=torch.load(ROOT/'data'/ds/'graph.pt',weights_only=True,map_location='cpu')
  torch.save(features[[lookup[i] for i in graph['external_prime_indices']]].clone(),ROOT/'data'/ds/'coder_features.pt')
 audit=dict(status='passed',device='cpu',total_entities=len(ids),audited_fixed_inputs_reused=len(reused),newly_encoded=len(missing),
  model=manifest,pooling='pooler_output + L2',input='official PrimeKG node_name only',truncated=truncated,
  unit_norm_max_error=float((features.norm(dim=-1)-1).abs().max()),old_neural_checkpoint_loaded=False,
  feature_sha256={ds:sha(ROOT/'data'/ds/'coder_features.pt') for ds in ['ALMANAC','ONEIL']},seconds=time.monotonic()-start)
 write(ROOT/'artifacts/ENCODING_AUDIT.json',audit);write(ROOT/'artifacts/encoding_progress.json',dict(status='complete',new_encoded=len(missing),total_new=len(missing),seconds=time.monotonic()-start))
 print(audit,flush=True)
if __name__=='__main__':main()
