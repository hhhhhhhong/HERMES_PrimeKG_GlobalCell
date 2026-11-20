"""Label-free BRICS fragment graphs, preserving occurrences and cut bonds."""
from collections import Counter

import torch
from rdkit import Chem, rdBase
from rdkit.Chem import BRICS


ATOM_CARDINALITIES = (120, 12, 12, 9, 9, 10, 2, 2, 258)
BOND_CARDINALITIES = (6, 9, 2, 2)
BOND_TYPES = ('SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC', 'DATIVE')


def atom_features(atom):
    charge = atom.GetFormalCharge()
    return [min(atom.GetAtomicNum(), 119), min(atom.GetTotalDegree(), 11),
            charge + 5 if -5 <= charge <= 5 else 11,
            min(int(atom.GetHybridization()), 8), min(int(atom.GetChiralTag()), 8),
            min(atom.GetTotalNumHs(), 9), int(atom.GetIsAromatic()),
            int(atom.IsInRing()), min(atom.GetIsotope(), 257)]


def bond_features(bond):
    kind = str(bond.GetBondType())
    return [BOND_TYPES.index(kind) if kind in BOND_TYPES else 5,
            min(int(bond.GetStereo()), 8), int(bond.GetIsConjugated()),
            int(bond.IsInRing())]


def decompose_brics(smiles, drug_id='drug'):
    """Return terminal fragments plus the original bonds between fragments.

    BRICSDecompose returns a set and discards repeated fragment occurrences.
    Breaking bonds and enumerating connected components preserves multiplicity.
    Disconnected salts/components are retained. Nothing is silently truncated.
    """
    molecule = Chem.MolFromSmiles(str(smiles))
    if molecule is None or molecule.GetNumAtoms() == 0:
        raise ValueError('invalid or empty SMILES for ' + str(drug_id))
    canonical = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    molecule = Chem.MolFromSmiles(canonical)
    cuts = list(BRICS.FindBRICSBonds(molecule, randomizeOrder=False))
    broken = BRICS.BreakBRICSBonds(molecule, bonds=cuts)
    mappings = []
    fragments = Chem.GetMolFrags(broken, asMols=True, sanitizeFrags=True,
                                fragsMolAtomMapping=mappings)
    records = sorted(zip(fragments, mappings), key=lambda item:
                     Chem.MolToSmiles(item[0], canonical=True, isomericSmiles=True))
    owner = {}
    for number, (_, mapping) in enumerate(records):
        for atom in mapping:
            if atom < molecule.GetNumAtoms():
                owner[atom] = number
    if len(owner) != molecule.GetNumAtoms():
        raise ValueError('BRICS lost original atoms for ' + str(drug_id))
    if sum(fragment.GetNumHeavyAtoms() for fragment, _ in records) != molecule.GetNumHeavyAtoms():
        raise ValueError('BRICS heavy atom count changed for ' + str(drug_id))
    connections = []
    for (left, right), attachment in cuts:
        bond = molecule.GetBondBetweenAtoms(left, right)
        source, target = owner[left], owner[right]
        if source == target:
            raise ValueError('BRICS cut did not separate fragments for ' + str(drug_id))
        connections.append(dict(source=source, target=target,
            original_atom_pair=[left, right], attachment_types=list(attachment),
            bond_features=bond_features(bond)))
    names = [Chem.MolToSmiles(fragment, canonical=True, isomericSmiles=True)
             for fragment, _ in records]
    metadata = dict(drug_id=str(drug_id), input_smiles=str(smiles),
        canonical_smiles=canonical, fragments=names, fragment_count=len(names),
        unique_fragment_count=len(set(names)), repeated_fragment_counts=dict(Counter(names)),
        heavy_atom_count=molecule.GetNumHeavyAtoms(), original_atom_count=molecule.GetNumAtoms(),
        fragment_original_atom_ids=[[atom for atom in mapping if atom < molecule.GetNumAtoms()]
                                    for _, mapping in records],
        fragment_heavy_atom_counts=[fragment.GetNumHeavyAtoms() for fragment, _ in records],
        cuts=connections, uncut_whole_molecule=len(cuts) == 0 and len(records) == 1)
    # Canonical fragment atom order also makes chirality categories consistent
    # when the same fragment occurs inside different parent molecules.
    return [Chem.MolFromSmiles(name) for name in names], connections, metadata


def build_brics_corpus(drug_ids, smiles_by_id):
    drug_ids = list(map(str, drug_ids))
    if not drug_ids or len(drug_ids) != len(set(drug_ids)):
        raise ValueError('expected a nonempty, unique ordered drug list')
    atoms, atom_edges, atom_bonds, atom_fragment = [], [], [], []
    fragment_drug, fragment_heavy, fragment_edges, fragment_bonds = [], [], [], []
    metadata = []
    for drug_number, drug_id in enumerate(drug_ids):
        if drug_id not in smiles_by_id:
            raise ValueError('missing SMILES for ' + drug_id)
        fragments, cuts, record = decompose_brics(smiles_by_id[drug_id], drug_id)
        first_fragment = len(fragment_drug)
        for fragment in fragments:
            fragment_number = len(fragment_drug)
            offset = len(atoms)
            atoms.extend(atom_features(atom) for atom in fragment.GetAtoms())
            atom_fragment.extend([fragment_number] * fragment.GetNumAtoms())
            for bond in fragment.GetBonds():
                a, b = offset + bond.GetBeginAtomIdx(), offset + bond.GetEndAtomIdx()
                atom_edges.extend([(a, b), (b, a)])
                atom_bonds.extend([bond_features(bond), bond_features(bond)])
            fragment_drug.append(drug_number)
            fragment_heavy.append(fragment.GetNumHeavyAtoms())
        for cut in cuts:
            a, b = first_fragment + cut['source'], first_fragment + cut['target']
            fragment_edges.extend([(a, b), (b, a)])
            fragment_bonds.extend([cut['bond_features'], cut['bond_features']])
        metadata.append(record)
    edge_tensor = lambda values: torch.tensor(values, dtype=torch.long).reshape(-1, 2).t().contiguous()
    bond_tensor = lambda values: torch.tensor(values, dtype=torch.long).reshape(-1, len(BOND_CARDINALITIES))
    tensors = dict(atoms=torch.tensor(atoms, dtype=torch.long),
        atom_edges=edge_tensor(atom_edges), atom_bonds=bond_tensor(atom_bonds),
        atom_fragment=torch.tensor(atom_fragment, dtype=torch.long),
        fragment_drug=torch.tensor(fragment_drug, dtype=torch.long),
        fragment_heavy=torch.tensor(fragment_heavy, dtype=torch.float32),
        fragment_edges=edge_tensor(fragment_edges), fragment_bonds=bond_tensor(fragment_bonds),
        drug_heavy=torch.tensor([m['heavy_atom_count'] for m in metadata], dtype=torch.float32),
        drug_fragment_counts=torch.bincount(torch.tensor(fragment_drug), minlength=len(drug_ids)).float())
    for values, cardinalities in [(tensors['atoms'], ATOM_CARDINALITIES),
                                  (tensors['atom_bonds'], BOND_CARDINALITIES),
                                  (tensors['fragment_bonds'], BOND_CARDINALITIES)]:
        for column, limit in enumerate(cardinalities):
            if values.numel() and not ((values[:, column] >= 0) & (values[:, column] < limit)).all():
                raise ValueError('categorical chemical feature outside vocabulary')
    audit = dict(schema_version=1, method='BRICS terminal fragments with original cut-bond graph',
        rdkit_version=rdBase.rdkitVersion, drug_ids=drug_ids, drug_count=len(drug_ids),
        fragment_occurrences=len(fragment_drug),
        unique_fragments=len({f for m in metadata for f in m['fragments']}),
        atom_nodes=len(atoms), directed_atom_edges=len(atom_edges),
        directed_fragment_edges=len(fragment_edges),
        uncut_whole_molecule_count=sum(m['uncut_whole_molecule'] for m in metadata),
        drugs_with_repeated_fragments=sum(m['fragment_count'] > m['unique_fragment_count'] for m in metadata),
        fragment_count_range=[min(m['fragment_count'] for m in metadata), max(m['fragment_count'] for m in metadata)],
        labels_used=False, learned_vocabulary=False, atoms_preserved=True,
        duplicates_preserved=True, fragment_truncation=False,
        disconnected_components='retained, no fabricated connections',
        original_cut_bond_stereo_preserved=True, drugs=metadata)
    return tensors, audit
