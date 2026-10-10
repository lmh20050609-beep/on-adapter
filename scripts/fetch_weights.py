"""Download the frozen text-encoder weights at the pinned revisions in weights/<name>/SOURCE.json."""
import json
from pathlib import Path
from huggingface_hub import snapshot_download

ROOT = Path(__file__).resolve().parents[1]
for source in sorted((ROOT / 'weights').glob('*/SOURCE.json')):
    meta = json.loads(source.read_text('utf-8'))
    snapshot_download(repo_id=meta['repo_id'], revision=meta['revision'], local_dir=source.parent)
    print('downloaded', meta['repo_id'], '@', meta['revision'], '->', source.parent)
