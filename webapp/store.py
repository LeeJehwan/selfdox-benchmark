"""Where the site keeps its state: a private Hugging Face dataset in production (HubStore),
a folder in tests (LocalStore). Paths are relative, e.g. "teams.json" or
"files/<team slug>/<submission id>/agent.zip"."""
import json
import shutil
import tempfile
from pathlib import Path


def _checked(path):
    """A store path is relative and stays inside the store: no absolute paths, no '..'."""
    p = Path(path)
    if p.is_absolute() or ".." in p.parts:
        raise ValueError(f"store path must be relative and inside the store: {path!r}")
    return p


class LocalStore:
    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def read_json(self, path, default=None):
        p = self.root / _checked(path)
        return json.loads(p.read_text()) if p.exists() else default

    def write_json(self, path, obj, message=""):
        p = self.root / _checked(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1))
        tmp.replace(p)

    def write_jsons(self, files, message=""):
        for path in files:
            _checked(path)                      # refuse any bad path before writing anything
        for path, obj in files.items():
            self.write_json(path, obj, message)

    def put_file(self, local_path, path, message=""):
        p = self.root / _checked(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, p)

    def get_file(self, path, dest):
        shutil.copyfile(self.root / _checked(path), dest)
        return Path(dest)


class HubStore:
    """A private dataset repository on the Hugging Face Hub; every write is one commit."""

    def __init__(self, repo_id, token=None):
        from huggingface_hub import HfApi
        self.api, self.repo, self.token = HfApi(token=token), repo_id, token

    def _fetch(self, path, use):
        """Download one file into a temporary cache, hand it to `use` while it exists, then
        delete the cache. Nothing is left behind, however often this runs."""
        from huggingface_hub import hf_hub_download
        with tempfile.TemporaryDirectory() as cache:
            local = hf_hub_download(self.repo, path, repo_type="dataset", token=self.token,
                                    force_download=True, cache_dir=cache)
            return use(Path(local))

    def read_json(self, path, default=None):
        from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
        try:
            return self._fetch(path, lambda p: json.loads(p.read_text()))
        except LocalEntryNotFoundError:
            # the Hub was unreachable: this is an outage, not a missing file; never report
            # the default, or the caller would overwrite real state with it
            raise
        except EntryNotFoundError:
            # the remote file genuinely does not exist
            return default

    def write_json(self, path, obj, message=""):
        data = json.dumps(obj, ensure_ascii=False, indent=1).encode()
        self.api.upload_file(path_or_fileobj=data, path_in_repo=path, repo_id=self.repo,
                             repo_type="dataset", commit_message=message or f"update {path}")

    def write_jsons(self, files, message=""):
        """Several JSON files in one commit."""
        from huggingface_hub import CommitOperationAdd
        ops = [CommitOperationAdd(str(_checked(path)),
                                  json.dumps(obj, ensure_ascii=False, indent=1).encode())
               for path, obj in files.items()]
        self.api.create_commit(self.repo, repo_type="dataset", operations=ops,
                               commit_message=message or "update state")

    def put_file(self, local_path, path, message=""):
        self.api.upload_file(path_or_fileobj=str(local_path), path_in_repo=path,
                             repo_id=self.repo, repo_type="dataset",
                             commit_message=message or f"add {path}")

    def get_file(self, path, dest):
        self._fetch(path, lambda p: shutil.copyfile(p, dest))
        return Path(dest)
