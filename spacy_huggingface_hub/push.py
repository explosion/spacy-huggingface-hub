from typing import Optional, Union, Dict, Any, List
import typer
import zipfile
import shutil
import json
import yaml
from huggingface_hub import Repository, HfApi, HfFolder
from pathlib import Path
from wasabi import Printer

# Allow using the package without spaCy being installed
try:
    from spacy.cli._util import app as spacy_cli
except ImportError:
    spacy_cli = None

TOKEN_CLASSIFICATION_COMPONENTS = ["ner", "tagger", "morphologizer"]
TEXT_CLASSIFICATION_COMPONENTS = ["textcat", "textcat_multilabel"]

SPACY_HF_HUB_HELP = """CLI for uploading spaCy pipelines to the
Hugging Face Hub (https://huggingface.co). Takes .whl files packaged with
`spacy package` with `--build wheel` and takes care of auto-generating all
meta information.
"""

NAME = "spacy"
HELP = """spaCy Command-line Interface
DOCS: https://spacy.io/api/cli
"""


hf_hub_cli = typer.Typer(
    name="huggingface-hub", help=SPACY_HF_HUB_HELP, no_args_is_help=True
)
# Add the subcommand to spaCy's CLI if spaCy is available
if spacy_cli is not None:
    spacy_cli.add_typer(hf_hub_cli)


@hf_hub_cli.command("push")
def huggingface_hub_push_cli(
    # fmt: off
    whl_path: Path = typer.Argument(..., help="Path to whl file", exists=True),
    organization: Optional[str] = typer.Option(None, "--org", "-o", help="Name of organization to which the pipeline should be uploaded"),
    commit_msg: str = typer.Option("Update spaCy pipeline", "--msg", "-m", help="Commit message to use for update"),
    local_repo_path: Path = typer.Option("hub", "--local-repo", "-l", help="Local path for creating repo"),
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Output additional info for debugging, e.g. the full generated hub metadata"),
    # fmt: on
):
    """
    Push a spaCy pipeline (.whl) to the Hugging Face Hub.
    """
    push(whl_path, organization, commit_msg, local_repo_path, verbose=verbose)


def push(
    whl_path: Union[str, Path],
    namespace: Optional[str] = None,
    commit_msg: str = "Update spaCy pipeline",
    local_repo_path: Union[Path, str] = "hub",
    *,
    silent: bool = False,
    verbose: bool = False,
) -> Dict[str, str]:
    msg = Printer(no_print=silent)
    whl_path = Path(whl_path)
    if not whl_path.exists():
        msg.fail(f"Can't find wheel path: {whl_path}")
    filename = whl_path.stem
    repo_name, version, _, _, _ = filename.split("-")
    versioned_name = repo_name + "-" + version
    repo_local_path = Path(local_repo_path) / repo_name

    # Create the repo (or clone its content if it's nonempty)
    api = HfApi()
    repo_url = api.create_repo(
        name=repo_name,
        token=HfFolder.get_token(),
        organization=namespace,
        # TODO: Can we support private packages as well via a flag?
        private=False,
        exist_ok=True,
    )
    repo = Repository(repo_local_path, clone_from=repo_url)
    repo.git_pull(rebase=True)
    # TODO: Are there other files we need to add here?
    repo.lfs_track(["*.whl", "*.npz", "*strings.json", "vectors"])
    info_msg = f"Publishing to repository '{repo_name}'"
    if namespace is not None:
        info_msg += f" ({namespace})"
    msg.info(info_msg)

    # Extract information from whl file
    with zipfile.ZipFile(whl_path, "r") as zip_ref:
        base_name = Path(repo_name) / versioned_name
        for file_name in zip_ref.namelist():
            if file_name.startswith(str(base_name)):
                zip_ref.extract(file_name, local_repo_path)
    msg.good("Extracted information from .whl file")

    # Move files up one directory
    extracted_dir = repo_local_path / versioned_name
    for filename in extracted_dir.iterdir():
        dst = repo_local_path / filename.name
        if dst.is_dir():
            shutil.rmtree(str(dst))
        elif dst.is_file():
            dst.unlink()
        shutil.move(str(filename), str(dst))
    shutil.rmtree(str(extracted_dir))

    # Create model card, including HF tags
    metadata = _create_model_card(repo_name, repo_local_path)
    msg.good("Created model card")
    msg.text(f"{repo_name} (v{version})")
    if verbose:
        print(metadata)

    # Remove version from whl filename
    dst_file = repo_local_path / f"{repo_name}-any-py3-none-any.whl"
    shutil.copyfile(str(whl_path), str(dst_file))

    msg.text("Pushing repository to the hub...")
    url = repo.push_to_hub(commit_message=commit_msg)
    url, _ = url.split("/commit/")
    msg.good(f"Pushed repository '{repo_name}' to the hub")
    whl_url = f"{url}/resolve/main/{repo_name}-any-py3-none-any.whl"
    if not silent:
        print(f"\nView your model here:\n{url}\n")
        print(f"Install your model:\npip install {whl_url}")
    return {"url": url, "whl_url": whl_url}


def _create_model_card(repo_name: str, repo_dir: Path) -> Dict[str, Any]:
    meta_path = repo_dir / "meta.json"
    with meta_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    lang = data["lang"] if data["lang"] != "xx" else "multilingual"
    lic = data.get("license", "").replace(" ", "-")
    tags = ["spacy"]
    for component in data["components"]:
        if (
            component in TOKEN_CLASSIFICATION_COMPONENTS
            and "token-classification" not in tags
        ):
            tags.append("token-classification")
        if (
            component in TEXT_CLASSIFICATION_COMPONENTS
            and "text-classification" not in tags
        ):
            tags.append("text-classification")

    metadata = _insert_values_as_list({}, "tags", tags)
    metadata = _insert_values_as_list(metadata, "language", lang)
    metadata = _insert_value(metadata, "license", lic)
    metadata["model-index"] = _create_model_index(repo_name, data["performance"])
    metadata = yaml.dump(metadata, sort_keys=False)
    metadata_section = f"---\n{metadata}---\n"

    # Read README generated by package
    readme_path = repo_dir / "README.md"
    readme = ""
    if readme_path.exists():
        with readme_path.open("r", encoding="utf8") as f:
            readme = f.read()
    with readme_path.open("w", encoding="utf-8") as f:
        f.write(metadata_section)
        f.write(readme)
    return metadata


def _insert_value(
    metadata: Dict[str, Any], name: str, value: Optional[Any]
) -> Dict[str, Any]:
    if value is None or value == "":
        return metadata
    metadata[name] = value
    return metadata


def _insert_values_as_list(
    metadata: Dict[str, Any], name: str, values: Optional[Any]
) -> Dict[str, List[Any]]:
    if values is None:
        return metadata
    if isinstance(values, str):
        values = [values]
    if len(values) == 0:
        return metadata
    metadata[name] = list(values)
    return metadata


def _create_metric(name: str, t: str, value: float) -> Dict[str, Union[str, float]]:
    return {"name": name, "type": t, "value": value}


def _create_p_r_f_list(
    precision: float, recall: float, f_score: float
) -> List[Dict[str, Union[str, float]]]:
    precision = _create_metric("Precision", "precision", precision)
    recall = _create_metric("Recall", "recall", recall)
    f_score = _create_metric("F Score", "f_score", f_score)
    return [precision, recall, f_score]


def _create_model_index(repo_name: str, data: Dict[str, Any]) -> Dict[str, Any]:
    # TODO: add some more metrics here
    model_index = {"name": repo_name}
    results = []
    if "ents_p" in data:
        results.append(
            {
                "tasks": {
                    "name": "NER",
                    "type": "token-classification",
                    "metrics": _create_p_r_f_list(
                        data["ents_p"], data["ents_r"], data["ents_f"]
                    ),
                }
            }
        )
    if "tag_acc" in data:
        results.append(
            {
                "tasks": {
                    "name": "POS",
                    "type": "token-classification",
                    "metrics": [
                        _create_metric("Accuracy", "accuracy", data["tag_acc"])
                    ],
                }
            }
        )
    if "sents_p" in data:
        results.append(
            {
                "tasks": {
                    "name": "SENTER",
                    "type": "token-classification",
                    "metrics": _create_p_r_f_list(
                        data["sents_p"], data["sents_r"], data["sents_f"]
                    ),
                }
            }
        )
    if "dep_uas" in data:
        results.append(
            {
                "tasks": {
                    "name": "UNLABELED_DEPENDENCIES",
                    "type": "token-classification",
                    "metrics": [
                        _create_metric("Accuracy", "accuracy", data["dep_uas"])
                    ],
                }
            }
        )
    if "dep_las" in data:
        results.append(
            {
                "tasks": {
                    "name": "LABELED_DEPENDENCIES",
                    "type": "token-classification",
                    "metrics": [
                        _create_metric("Accuracy", "accuracy", data["dep_uas"])
                    ],
                }
            }
        )
    model_index["results"] = results
    return model_index
