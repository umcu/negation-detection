import json
import pandas as pd

def evaluate_per_example(annotation_file, meta_cat, result_column_name):
    """Evaluates metacat on annotation file per example, and include example ID.
    
    This code predicts the negation for every example in an annotation file (MedCAT Trainer JSON format).
    For every example custom Span and Doc objects are created to provide data in the expected format for MetaCAT. This allows for using
    the tokens as they are annotated in the labeled data. SpaCy's Span and Doc objects create entities from tokenized
    text, which could be different from how human annotators tokenize text.
    This code is based on MedCAT's json_to_fake_spacy(), see:
    https://github.com/CogStack/MedCAT/blob/bbb2dc8aa452d0561709993078ce4f0297a63ff6/medcat/utils/meta_cat/data_utils.py#L133
    """
    class Empty(object):
        def __init__(self):
            pass

    class CustomSpan(object):
        def __init__(self, start_char, end_char, id):
            self._ = Empty()
            self.start_char = start_char
            self.end_char = end_char
            self._.id = id
            self._.meta_anns = None

    class CustomDoc(object):
        def __init__(self, text, id):
            self._ = Empty()
            self._.share_tokens = None
            self.ents = []
            self._ents = self.ents
            self.text = text
            self.id = id

    # Create empty list to store predictions for each entity
    predictions = []
    
    # Load annotated data
    with open(annotation_file) as f:
        annotations = json.load(f)

    # Loop over every document
    for document in annotations['projects'][0]['documents']:
        document_name = document['name']
        text = document['text']
        doc = CustomDoc(text=text, id=document_name)

        # Loop over every annotated entity
        for annotation in document['annotations']:

            # Extract data
            start_char = annotation['start']
            end_char = annotation['end']

            # Create custom ID
            entity_id = f'{document_name}_{start_char}_{end_char}'

            # Add entity as custom Span to custom Doc object
            doc.ents.append(CustomSpan(start_char, end_char, entity_id))

        doc = meta_cat(doc)

        # Retrieve predictions
        for ent in doc.ents:
            entity_id = ent._.id
            annotation = ent._.meta_anns['Negation']['value']
            predictions.append([entity_id, annotation])

    return pd.DataFrame(predictions, columns=['entity_id', result_column_name])
