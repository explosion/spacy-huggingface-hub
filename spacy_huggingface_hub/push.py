from typing import Optional, Union, Dict, Any, List
import typer
import zipfile
import shutil
import json
import yaml
from huggingface_hub import HfApi, upload_folder, whoami
from pathlib import Path
import tempfile
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
    verbose: bool = typer.Option(False, "--verbose", "-V", help="Output additional info for debugging, e.g. the full generated hub metadata"),
    # fmt: on
):
    """
    Push a spaCy pipeline (.whl) to the Hugging Face Hub.
    """
    push(whl_path, organization, commit_msg, verbose=verbose)


def push(
    whl_path: Union[str, Path],
    namespace: Optional[str] = None,
    commit_msg: str = "Update spaCy pipeline",
    *,
    silent: bool = False,
    verbose: bool = False,
) -> Dict[str, str]:
    msg = Printer(no_print=silent)
    whl_path = Path(whl_path)
    if not whl_path.exists():
        msg.fail(f"Can't find wheel path: {whl_path}", exits=1)
    if whl_path.suffix != ".whl":
        msg.fail(
            f"Not a valid .whl file: {whl_path}. Make sure to run `spacy "
            "package` with `--build wheel` to generate a wheel for you pipeline.",
            exits=1,
        )
    filename = whl_path.stem
    repo_name, version, _, _, _ = filename.split("-")

    if namespace is None:
        namespace = whoami()["name"]
    repo_id = f"{namespace}/{repo_name}"

    # Create the repo (or clone its content if it's nonempty)
    HfApi().create_repo(
        repo_id=repo_id,
        # TODO: Can we support private packages as well via a flag?
        private=False,
        exist_ok=True,
    )
    msg.info(f"Publishing to repository '{repo_id}'")

    with tempfile.TemporaryDirectory() as tmpdirname:
        repo_local_path = Path(tmpdirname) / repo_name
        # Extract information from whl file
        with zipfile.ZipFile(whl_path, "r") as zip_ref:
            for file_name in zip_ref.namelist():
                if Path(file_name) != Path(repo_name) / "__init__.py":
                    print("result", zip_ref.extract(file_name, tmpdirname), file_name)
        msg.good("Extracted information from .whl file")

        # Some whl files might not have the version in the name (such as the ones at HF)
        # but the version can be found in the meta.json file.
        if version == "any":
            version = json.load(open(Path(tmpdirname) / repo_name / "meta.json"))["version"]

        # Move files up one directory
        # The original structure is ca_core/ca_core-version/files. files are moved one level
        # up.
        versioned_name = repo_name + "-" + version
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
        url = upload_folder(
            repo_id=repo_id,
            folder_path=repo_local_path,
            path_in_repo="",
            commit_message=commit_msg,
        )
        url, _ = url.split("/tree/")
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

    lic = data.get("license", "").replace(" ", "-").lower()
    # HF accepts gpl-3.0 directly
    lic = lic.replace("gnu-", "")

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
    metric_name: str, precision: float, recall: float, f_score: float
) -> List[Dict[str, Union[str, float]]]:
    precision = _create_metric(f"{metric_name} Precision", "precision", precision)
    recall = _create_metric(f"{metric_name} Recall", "recall", recall)
    f_score = _create_metric(f"{metric_name} F Score", "f_score", f_score)
    return [precision, recall, f_score]


def _create_model_index(repo_name: str, data: Dict[str, Any]) -> List[Dict[str, Any]]:
    # TODO: add some more metrics here
    model_index = {"name": repo_name}
    results = []
    if "ents_p" in data:
        results.append(
            {
                "task": {"name": "NER", "type": "token-classification"},
                "metrics": _create_p_r_f_list(
                    "NER", data["ents_p"], data["ents_r"], data["ents_f"]
                ),
            }
        )
    if "tag_acc" in data:
        results.append(
            {
                "task": {"name": "TAG", "type": "token-classification"},
                "metrics": [
                    _create_metric("TAG (XPOS) Accuracy", "accuracy", data["tag_acc"])
                ],
            }
        )
    if "pos_acc" in data:
        results.append(
            {
                "task": {"name": "POS", "type": "token-classification"},
                "metrics": [
                    _create_metric("POS (UPOS) Accuracy", "accuracy", data["pos_acc"])
                ],
            }
        )
    if "morph_acc" in data:
        results.append(
            {
                "task": {"name": "MORPH", "type": "token-classification"},
                "metrics": [
                    _create_metric(
                        "Morph (UFeats) Accuracy", "accuracy", data["morph_acc"]
                    )
                ],
            }
        )
    if "lemma_acc" in data:
        results.append(
            {
                "task": {"name": "LEMMA", "type": "token-classification"},
                "metrics": [
                    _create_metric("Lemma Accuracy", "accuracy", data["lemma_acc"])
                ],
            }
        )
    if "dep_uas" in data:
        results.append(
            {
                "task": {
                    "name": "UNLABELED_DEPENDENCIES",
                    "type": "token-classification",
                },
                "metrics": [
                    _create_metric(
                        "Unlabeled Attachment Score (UAS)", "f_score", data["dep_uas"]
                    )
                ],
            }
        )
    if "dep_las" in data:
        results.append(
            {
                "task": {
                    "name": "LABELED_DEPENDENCIES",
                    "type": "token-classification",
                },
                "metrics": [
                    _create_metric(
                        "Labeled Attachment Score (LAS)", "f_score", data["dep_las"]
                    )
                ],
            }
        )
    if "sents_p" in data:
        results.append(
            {
                "task": {"name": "SENTS", "type": "token-classification"},
                "metrics": [
                    _create_metric(
                        "Sentences F-Score", "f_score", data["sents_f"]
                    ),
                ],
            }
        )
    model_index["results"] = results
    return [model_index]
