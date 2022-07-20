<a href="https://explosion.ai"><img src="https://explosion.ai/assets/img/logo.svg" width="125" height="125" align="right" /></a>

# spacy-huggingface-hub: Push your spaCy pipelines to the Hugging Face Hub

This package provides a CLI command for uploading any trained spaCy pipeline packaged with [`spacy package`](https://spacy.io/api/cli#package) to the [Hugging Face Hub](https://huggingface.co/). It auto-generates all meta information for you, uploads a pretty README (requires spaCy v3.1+) and handles version control under the hood.

[![PyPi](https://img.shields.io/pypi/v/spacy-huggingface-hub.svg?style=flat-square&logo=pypi&logoColor=white)](https://pypi.python.org/pypi/spacy-huggingface-hub)
[![GitHub](https://img.shields.io/github/release/explosion/spacy-huggingface-hub/all.svg?style=flat-square&logo=github)](https://github.com/explosion/spacy-huggingface-hub/releases)

## 🤗 About the Hugging Face Hub

The [Hugging Face Hub](https://huggingface.co/) hosts Git-based repositories which are storage spaces that can contain all your files. These repositories have multiple advantages: **versioning** (commit history and diffs), **branches**, useful **metadata** about their tasks, languages, metrics and more, browser-based **visualizers** to explore the models interactively in your browser, as well as an **API** to use the models in production.

## 🚀 Quickstart

You can install `spacy-huggingface-hub` from pip:

```bash
pip install spacy-huggingface-hub
```

To check if the command has been registered successfully:

```bash
python -m spacy huggingface-hub --help
```


You can upload any pipeline packaged with [`spacy package`](https://spacy.io/api/cli#package). Make sure to set `--build wheel` to output a binary `.whl` file. The uploader will read all metadata from the pipeline package, including the auto-generated pretty `README.md` and the model details available in the `meta.json`.

```bash
huggingface-cli login
python -m spacy package ./en_ner_fashion ./output --build wheel
cd ./output/en_ner_fashion-0.0.0/dist
python -m spacy huggingface-hub push en_ner_fashion-0.0.0-py3-none-any.whl
```

The command will output two things:

- Where to find your repo in the Hub! For example, https://huggingface.co/spacy/en_core_web_sm
- And how to install the pipeline directly from the Hub!

```bash
pip install https://huggingface.co/spacy/en_core_web_sm/resolve/main/en_core_web_sm-any-py3-none-any.whl
```

Now you can share your pipelines very quickly with others. Additionally, you can also test your pipeline directly in the browser!

![Image of browser widget](https://user-images.githubusercontent.com/13643239/124529281-7e9a1b00-de0a-11eb-9069-093e3021a307.png)

## ⚙️ Usage and API

If spaCy is already installed in the same environment, this package automatically adds the `spacy huggingface-hub` commands to the CLI. If you don't have spaCy installed, you can also execute the CLI directly via the package.

### `push`

```bash
python -m spacy huggingface-hub push [whl_path] [--org] [--msg] [--local-repo] [--verbose]
```

```bash
python -m spacy_huggingface_hub push [whl_path] [--org] [--msg] [--local-repo] [--verbose]
```

| Argument             | Type         | Description                                                                                                                   |
| -------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------- |
| `whl_path`           | str / `Path` | The path to the `.whl` file packaged with [`spacy package`](https://spacy.io/api/cli#package).                                |
| `--org`, `-o`        | str          | Optional name of organization to which the pipeline should be uploaded.                                                       |
| `--msg`, `-m`        | str          | Commit message to use for update. Defaults to `"Update spaCy pipeline"`.                                                      |
| `--verbose`, `-V`    | bool         | Output additional info for debugging, e.g. the full generated hub metadata.                                                   |

### Usage from Python

Instead of using the CLI, you can also call the `push` function from Python. It returns a dictionary containing the `"url"` of the published model and the `"whl_url"` of the wheel file, which you can install with `pip install`

```python
from spacy_huggingface_hub import push

result = push("./en_ner_fashion-0.0.0-py3-none-any.whl")
print(result["url"])
```
