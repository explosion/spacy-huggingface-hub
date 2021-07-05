# spacy-huggingface-hub

🤗 Quickly push your spaCy pipelines to the [Hugging Face Hub](https://huggingface.co/)

## About the Hugging Face Hub

The Hugging Face Hub hosts Git-based repositories which are storage spaces that can contain all your files. These repositories have multiple advantages:

* Versioning (commit history and diffs).
* Branches.
* Useful metadata about their tasks, languages, metrics, and more.
* Anyone can play with the model directly in the browser!
* An API is provided to use the models in production settings.


## 🚀 Quickstart

You can install `spacy-huggingface_hub` from pip:

```
pip install spacy-huggingface_hub
```

To check if the command has been registered successfully:

```
python -m spacy huggingface_hub --help
```

Hugging Face uses Git Large File Storage to handle files larger than 10mb. You can find instructions on how to download it in https://git-lfs.github.com/.

You can then upload any packaged pipeline!

```
python -m spacy huggingface_hub push en_ner_fashion-0.0.0-py3-none-any.whl
```

The command will output two things
* Where to find your repo in the Hub! For example, https://huggingface.co/spacy/en_core_web_sm
* And how to install the pipeline directly from the Hub!

```
pip install https://huggingface.co/osanseviero/en_ner_fashion/resolve/main/en_ner_fashion-any-py3-none-any.whl
```

Now you can share your pipelines very quickly with others. Additionally, you can also test your pipeline directly in the browser!

!["Image of browser widget.](assets/ner_on_the_hub.png)
