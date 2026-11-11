"""Download resumably and safely extract only the four original PCB directories."""
import argparse,json,shutil,subprocess,tarfile
from pathlib import Path,PurePosixPath
from .prepare import PCBS,sha
URL='https://amazon-visual-anomaly.s3.us-west-2.amazonaws.com/VisA_20220922.tar'
SIZE=1929840640

def download(destination,expected_sha=None):
    destination=Path(destination); destination.mkdir(parents=True,exist_ok=True)
    archive=destination.parent/'VisA_20220922.tar'
    if not archive.exists() or archive.stat().st_size!=SIZE:
        subprocess.run(['curl','--fail','--location','--retry','3','--continue-at','-','--output',str(archive),URL],check=True)
    assert archive.stat().st_size==SIZE, 'Archive size mismatch'
    digest=sha(archive)
    if expected_sha: assert digest==expected_sha,'Archive SHA256 mismatch'
    # No extractall: reject links and path traversal, copy selected regular files only.
    with tarfile.open(archive,'r:') as tar:
        for member in tar:
            path=PurePosixPath(member.name)
            if not path.parts or path.parts[0] not in PCBS:continue
            assert not path.is_absolute() and '..' not in path.parts
            assert member.isdir() or member.isfile(), 'Links or special files are forbidden'
            target=destination/path
            if member.isdir():target.mkdir(parents=True,exist_ok=True);continue
            target.parent.mkdir(parents=True,exist_ok=True)
            tmp=target.with_name(target.name+'.part')
            with tar.extractfile(member) as src,tmp.open('wb') as dst:shutil.copyfileobj(src,dst)
            assert tmp.stat().st_size==member.size
            tmp.replace(target)
    (destination/'archive_provenance.json').write_text(json.dumps({'url':URL,'bytes':SIZE,'sha256':digest},indent=2))
    print(f'Extracted pcb1-pcb4 to {destination}; SHA256={digest}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--destination',required=True);p.add_argument('--expected-sha');a=p.parse_args();download(a.destination,a.expected_sha)
