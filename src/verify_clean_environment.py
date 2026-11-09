
import os, subprocess, shutil, venv, json
from pathlib import Path
from src.project import ROOT

def main():
    clean=ROOT/'build/clean_room'; clean.mkdir(parents=True,exist_ok=True)
    for folder in ['src','tests','data']:
        shutil.copytree(ROOT/folder,clean/folder,dirs_exist_ok=True,ignore=shutil.ignore_patterns('__pycache__'))
    shutil.copy2(ROOT/'requirements.txt',clean/'requirements.txt')
    venv.EnvBuilder(with_pip=True).create(clean/'.venv')
    py=clean/'.venv/bin/python'
    env={**os.environ,'MPLCONFIGDIR':str(clean/'mplcache'),'PYTHONPATH':str(clean),'LOKY_MAX_CPU_COUNT':'1'}
    with (ROOT/'results/clean_environment.log').open('w') as log:
        for cmd in [[str(py),'-m','pip','install','-r','requirements.txt'],[str(py),'-m','src.run_experiments'],[str(py),'-m','pytest','-q']]:
            print('RUN',cmd,file=log,flush=True)
            subprocess.run(cmd,cwd=clean,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
    import pandas as pd
    a=pd.read_csv(ROOT/'results/tables/model_comparison.csv'); b=pd.read_csv(clean/'results/tables/model_comparison.csv')
    pd.testing.assert_frame_equal(a,b,atol=1e-10,rtol=1e-10)
    (ROOT/'results/clean_environment_verified.json').write_text(json.dumps({'isolated_venv':True,'requirements_install':True,'experiments_ran':True,'tests_passed':True,'metrics_match':True,'tolerance':1e-10},indent=2))
    print('Clean environment install, full pipeline, tests and metric agreement verified.')
if __name__=='__main__':main()
