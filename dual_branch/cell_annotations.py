"""Public Cellosaurus snapshots with exact, unique human-name matching only."""
import concurrent.futures
import csv
import hashlib
import json
import urllib.parse
import urllib.request
from .common import ROOT, DISPATCH, DATA
from global_cell.common import write


def canonical(value):
    return ''.join(c for c in value.casefold() if c.isalnum())


def fetch(name):
    parameters = dict(q='idsy:"%s" ox:9606' % name, format='json', rows=1000,
        fields='id,sy,ac,ox,di,derived-from-site,cell-type,ca')
    url = 'https://api.cellosaurus.org/search/cell-line?' + urllib.parse.urlencode(parameters)
    path = DISPATCH / 'cellosaurus_raw' / (hashlib.sha256(name.encode()).hexdigest() + '.json')
    try:
        if path.exists():
            raw = path.read_bytes()
        else:
            with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'research-input-audit'}), timeout=20) as response:
                raw = response.read()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(raw)
        payload = json.loads(raw)
        candidates = payload.get('Cellosaurus', {}).get('cell-line-list', [])
        found = {}
        for candidate in candidates:
            names = [x['value'] for x in candidate.get('name-list', [])]
            if not any(canonical(n) == canonical(name) for n in names):
                continue
            if not any(str(s.get('accession')) == '9606' for s in candidate.get('species-list', [])):
                continue
            accession = next(x['value'] for x in candidate['accession-list'] if x['type'] == 'primary')
            found[accession] = candidate
        record = dict(input_name=name, query_url=url, raw_sha256=hashlib.sha256(raw).hexdigest(),
            snapshot=str(path), match_rule='unique human identifier/synonym after case/punctuation normalization',
            candidate_accessions=sorted(found), status='unique' if len(found) == 1 else 'unresolved' if not found else 'ambiguous')
        if len(found) == 1:
            accession, candidate = next(iter(found.items()))
            record.update(accession=accession, source_url='https://www.cellosaurus.org/' + accession,
                official_name=next(n['value'] for n in candidate['name-list'] if n['type'] == 'identifier'),
                disease_labels=sorted({x['label'] for x in candidate.get('disease-list', [])}),
                tissue_labels=sorted({x['site']['value'] for x in candidate.get('derived-from-site-list', [])}),
                cell_type_labels=sorted({x.get('value', '') for x in candidate.get('cell-type-list', []) if x.get('value')}),
                category=candidate.get('category'))
        return name, record
    except Exception as error:
        return name, dict(input_name=name, status='unavailable', query_url=url, error=repr(error))


def main():
    names = set()
    for dataset in ['ALMANAC', 'ONEIL']:
        path = ROOT / 'vendor/hermes/data' / (dataset + '-COSMIC') / 'cell line_gene_expression.csv'
        with path.open() as stream:
            names.update(row[0] for row in list(csv.reader(stream))[1:])
    records = {}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        for name, record in pool.map(fetch, sorted(names)):
            records[name] = record
            print(name, record['status'], record.get('accession', ''), flush=True)
    write(DATA / 'cell_annotations.json', records)
    write(DISPATCH / 'CELL_ANNOTATION_AUDIT.json', dict(status='complete', names=len(names),
        unique_matches=sum(x['status'] == 'unique' for x in records.values()),
        unresolved=[name for name, r in records.items() if r['status'] != 'unique'],
        exact_match_only=True, guessed_matches=False, synergy_labels_used=False,
        source='https://api.cellosaurus.org/', annotations=str(DATA / 'cell_annotations.json')))


if __name__ == '__main__':
    main()
