
import argparse, hashlib, json, zipfile
from pathlib import Path
from src.project import ROOT, original_hashes

def package(group='GROUP_UNCONFIRMED'):
    if not group.replace('_','').replace('-','').isalnum():
        raise ValueError('Use letters, digits, underscores or hyphens for group')
    original=json.loads((ROOT/'results/original_manifest.json').read_text())
    assert original_hashes()==original, 'Original files changed'
    mandatory=['README.md','CHANGELOG.md','CONTRIBUTIONS.md','CONCLUSIONS.md','DEFENSE_QA.md','RUBRIC_CHECKLIST.md','requirements.txt','requirements-lock.txt',
    'notebooks/Endterm_IT_Vacancy_Seniority.ipynb','presentation/Endterm_IT_Vacancy_Seniority.pptx','presentation/SCRIPT.md','results/tables/model_comparison.csv','results/test_results.txt']
    assert all((ROOT/p).is_file() for p in mandatory)
    folders=['data','notebooks','src','results','presentation','tests']
    files=[ROOT/p for p in mandatory if '/' not in p]+[ROOT/'.gitignore']+list(ROOT.glob('*.pdf'))
    for folder in folders:
        files.extend(p for p in (ROOT/folder).rglob('*') if p.is_file() and not any(s in p.parts for s in ['__pycache__','.pytest_cache','.ipynb_checkpoints']) and not p.name.startswith(('.DS_Store','~$')))
    if (ROOT/'archive/midterm_originals.zip').exists():
        files.append(ROOT/'archive/midterm_originals.zip')
    files=sorted(set(files))
    hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    (ROOT/'SUBMISSION_MANIFEST.json').write_text(json.dumps(hashes,indent=2))
    archive=ROOT/f'Endterm_Zholdubayev_Diyar_{group}.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files+[ROOT/'SUBMISSION_MANIFEST.json']:
            z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for name,sha in hashes.items():
            assert hashlib.sha256(z.read(name)).hexdigest()==sha
    print(f'Archive: {archive.name}; {len(files)+1} files; all hashes verified')
    return archive
if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--group',default='GROUP_UNCONFIRMED'); args=p.parse_args(); package(args.group)
