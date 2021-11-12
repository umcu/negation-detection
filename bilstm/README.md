# biLSTM using MedCAT's MetaCAT functionality

## Installation
Install the requirements from the root directory of this repository.
```bash
pip install -r requirements_bilstm.txt
```

Downloading the spaCy model is currently commented out in the requirements file, because at the time of writing there is an issue with GitHub, see https://github.com/explosion/spaCy/issues/9606. The spaCy model is only required when using MedCAT together with MetaCAT (`04_medcat_usage.ipynb`).

When the service is fully functioning again, you could try installing the spaCy model using:
```bash
python -m spacy download nl_core_news_lg
``` 
