"""Global Cell/synergy propagation plus query-specific ordered path evidence.

Path extraction is offline and label-free. A small shared GRU encodes the
ordered entity/relation chain, preserving inverse relations and real endpoint
roles. Existing DrugPair x Cell gamma attends actual paths, alongside global
Cell/synergy channels. This is a task adaptation, not K-Paths author's model.
"""
import torch
from torch import nn
from .stable_prediction import StableTransformerModel, bounded_delta


class OrderedPathEncoder(nn.Module):
    def __init__(self, relation_count, dim=256):
        super().__init__()
        self.relation = nn.Embedding(relation_count, dim, padding_idx=0)
        self.sequence = nn.GRU(dim, dim, batch_first=True)

    def forward(self, nodes, node_ids, relation_ids, hops):
        entities = nodes[node_ids]
        relations = self.relation(relation_ids)
        sequence = torch.stack([entities[:,i//2] if i%2==0 else relations[:,i//2]
                                for i in range(9)], dim=1)
        hidden, _ = self.sequence(sequence)
        # Read the last REAL entity, before trailing padding. Path0 is a
        # masked sentinel whose value and derivative are strictly zero.
        result = hidden[torch.arange(len(hidden),device=hidden.device),2*hops]
        return result * (hops>0)[:,None]


class KPathsModel(StableTransformerModel):
    def __init__(self, data, dim=256):
        super().__init__(data, knowledge_mode='full', dim=dim)
        paths=data['paths']
        self.path_encoder=OrderedPathEncoder(len(paths['relation_names']),dim)
        for name in ['nodes','relations','hops','pair_paths','pair_mask']:
            self.register_buffer('path_'+name,paths[name].clone(),persistent=False)
        # Pure KG channels are not read as a global mixture: gamma receives
        # the exact query's paths. Preserve collaboration and Cell channels.
        keep=[]
        global_families={'Synergy_Hyperedge','cell_expression'}
        for i,key in enumerate(self.semantic.trace['token_keys']):
            if key[0]=='single' and key[1] in global_families:
                keep.append(i)
            elif key[0]=='interaction' and key[2] in global_families and key[4] in global_families:
                keep.append(i)
        assert keep,'Global Cell/synergy channels must be present'
        self.register_buffer('global_token_ids',torch.tensor(keep,dtype=torch.long),persistent=False)
        self.global_tokens=len(keep)
        self.global_rows=data['graph']['n_roots']*self.global_tokens
        self.selection_audit=dict(policy='target-anchored K-Paths, ordered query-specific gamma',
            graph=data['graph']['summary'],global_readout_families=sorted(global_families),
            global_readout_channels=self.global_tokens,paths=len(paths['hops'])-1,
            includes_Cell_and_fit_synergy=True,clinical_and_enzyme_paths_in_main_reader=False)

    def knowledge_table(self, base=None, capture=False):
        semantic=self.semantic
        slots,nodes=semantic.propagate(semantic.initial_nodes(),capture=capture or semantic.capture_propagation)
        global_table,_=semantic.interaction(slots,capture)
        path_table=self.path_encoder(nodes,self.path_nodes,self.path_relations,self.path_hops)
        # Shared trainer takes a single detached leaf table. Concatenating
        # both streams keeps its exact-full-fit, single-update semantics.
        return torch.cat([global_table[:,self.global_token_ids].reshape(-1,self.semantic.dim),path_table])

    def forward(self, rows, table, base, capture=False, use_knowledge=True, empty_evidence=False):
        ids=rows[:,:3].long().clone()
        ids[:,:2]=ids[:,:2].sort(dim=1).values
        own=base[ids]
        path_ids=self.path_pair_paths[ids[:,0]*self.num_drug+ids[:,1]]
        path_mask=self.path_pair_mask[ids[:,0]*self.num_drug+ids[:,1]]
        attention=None;gates=None
        if use_knowledge:
            global_table=table[:self.global_rows].reshape(-1,self.global_tokens,self.semantic.dim)
            global_available=self.semantic.interaction.mask[:,self.global_token_ids]
            global_values=global_table[ids]
            path_values=table[self.global_rows:][path_ids]
            evidence=torch.cat([global_values,path_values[:,None].expand(-1,3,-1,-1)],dim=2)
            available=torch.cat([global_available[ids],path_mask[:,None].expand(-1,3,-1)],dim=2)
            if empty_evidence:available=torch.zeros_like(available)
            context,attention=self.gamma(own,evidence,available,capture)
            _,raw_delta,gates=self.fusion(own,context)
            delta=bounded_delta(raw_delta,own,self.knowledge_fraction)
        else:
            context=torch.zeros_like(own);delta=raw_delta=torch.zeros_like(own)
            available=None
        prediction,tf_delta,tf_raw=self._decode_bounded(own+delta,own)
        detail=dict(own=own.detach(),delta=delta.detach(),raw_delta=raw_delta.detach(),
            transformer_delta=tf_delta.detach(),raw_transformer_delta=tf_raw.detach(),
            context=context.detach(),gates=gates.detach() if gates is not None else None,
            attention=attention,available=available.detach() if available is not None else None,
            path_ids=path_ids.detach(),path_mask=path_mask.detach(),
            global_tokens=self.global_tokens) if capture else None
        return prediction,detail

    def components(self):
        return dict(super().components(),path_encoder=list(self.path_encoder.parameters()))

    def optimizer_param_groups(self,base_lr=1e-4,knowledge_lr_mult=1.,weight_decay=1e-4):
        c=self.components()
        no_decay={'propagation_gates','fusion_output','prediction_transformer_output'}
        groups=[dict(name=name,params=ps,lr=base_lr,weight_decay=0. if name in no_decay else weight_decay)
                for name,ps in c.items()]
        actual=[id(p) for g in groups for p in g['params']]
        assert len(actual)==len(set(actual)) and set(actual)=={id(p) for p in self.parameters()}
        return groups
