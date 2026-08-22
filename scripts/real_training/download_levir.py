from pathlib import Path
import zipfile
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[2]
target=ROOT/"data"/"levir_cd"; target.mkdir(parents=True,exist_ok=True)

def safe_extract(archive: zipfile.ZipFile, destination: Path) -> None:
    """Reject archive members escaping the explicitly requested split directory."""
    base=destination.resolve()
    for member in archive.infolist():
        if not (base / member.filename).resolve().is_relative_to(base):
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)

for split in ("train","val","test"):
    archive=Path(hf_hub_download(repo_id="satellite-image-deep-learning/LEVIR-CD",
        filename=f"{split}.zip",repo_type="dataset",local_dir=target/"downloads"))
    out=target/split
    if not out.exists():
        out.mkdir(parents=True)
        with zipfile.ZipFile(archive) as z: safe_extract(z,out)
    print(split,archive.stat().st_size)
