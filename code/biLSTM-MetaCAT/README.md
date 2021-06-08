# medcat-pipeline
Contains code to produce MedCAT models used for text mining project.

## Installation

#### 1. Install MedCAT
To use this pipeline, you'll need to install MedCAT, which you can do with PIP, or clone it to test
custom changes.
 - PIP: 
```bash
pip install medcat==0.4.0.6

# If this fails due to because of a dependency on Torch, first install Torch:
pip install torch==1.4.0+cu92 torchvision==0.5.0+cu92 -f https://download.pytorch.org/whl/torch_stable.html
```

 - Git clone (This also requires you to add this directory to your PYTHONPATH or install it with `python -m pip install -e <path-to-medcat>`):
```bash
git clone git@github.com:CogStack/MedCAT.git
```
 
#### 2. Install required packages
Install the required Python Packages with PIP:
```bash
pip install -r requirements.txt
```

#### 3. Set MedCAT environment variables
MedCAT uses environment variables to set some parameters at runtime. You can download a file with all variables directly from https://github.com/CogStack/MedCAT/tree/master/envs or use it from the cloned MedCAT repository. Without passing environment variables at runtime (you can set them example in PyCharm, or system wide), defaults are used that are currently only documented in code. Some notable variables:

| Env variable | Description |
| ------------ | ----------- |
| SPACY_MODEL=en_core_sci_md | spaCy model, used for tokenization |
| MIN_CONCEPT_LENGTH=1 | Although it says minimum concept length, it is maximum length of concepts that are ignored |
| MIN_ACC=0.1  | Accuracy of link below this threshold is not shown as link in MedCAT Trainer |

#### 4. Download spaCy's trained pipeline
Download spaCy's trained pipeline, depending on which language model you want to use (configurable in the `env` file).

At the time of writing, the English scispacy model seems to work best with Dutch medical text.
```bash
# English, trained on biomedical data
pip install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.2.4/en_core_sci_md-0.2.4.tar.gz

# Dutch, trained on news
python -m spacy download nl_core_news_sm
```

## Data 

### Input: 
To create a MedCAT model, a Concept Database (CDB) and vocabulary (vocab) are needed. For this study, we used UMLS terms to create the CDB and the Dutch Medical Wikipedia for the vocabulary.   
To train a MetaCAT model for meta-annotations in addition to a MedCAT model, word embeddings and a tokenizer are required. For word embeddings as well as for the tokenizer the Dutch Medical Wikipedia is also used.  
Furthermore, a JSON file is required containing the text, the annotated concept, and the meta-annotations with labels.   

### Output:
MetaCAT returns as output a lstm.dat file which is trained on the EMC Dutch clinical corpus. One BiLSTM is trained on the full dataset. The other BiLSTMs are trained on one of the four specific data types.  
