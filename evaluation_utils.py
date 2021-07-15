import pandas as pd


def predict_negations(annotations, method, model):
    """
    Based on MedCAT trainer JSON format, predict negations.
    """
    result = []
    for document in annotations['projects'][0]['documents']:
        document_name = document['name']
        text = document['text']
        for annotation in document['annotations']:
            start = annotation['start']
            end = annotation['end']
            negation = annotation['meta_anns']['Negation']['value']
            if method == 'bilstm':
                prediction = model.predict_one(text, start, end)
            elif method == 'rule_based':
                print(f'{method}-method not implemented')
            elif method == 'roberta':
                print(f'{method}-method not implemented')
            else:
                print(f'{method}-method not implemented')

            result.append([f'{document_name}_{start}_{end}', negation, prediction])
    return pd.DataFrame(result, columns=['entity_id', 'annotation', method])


def print_statistics(result_df, method):
    """
    Calculate statistics
    """
    tp = result_df[(result_df.annotation == 'negated') & (result_df[method] == 'negated')].shape[0]
    tn = result_df[(result_df.annotation == 'not negated') & (result_df[method] == 'not negated')].shape[0]
    fp = result_df[(result_df.annotation == 'not negated') & (result_df[method] == 'negated')].shape[0]
    fn = result_df[(result_df.annotation == 'negated') & (result_df[method] == 'not negated')].shape[0]
    recall = round(tp / (tp + fn), 2)
    precision = round(tp / (tp + fp), 2)
    specificity = round(tn / (tn + fp), 2)
    accuracy = round((tp + tn) / (tp + fp + tn + fn), 2)
    f1 = round((2*tp) / ((2*tp) + fp + fn), 2)

    print(f'tp: {tp}')
    print(f'tn: {tn}')
    print(f'fp: {fp}')
    print(f'fn: {fn}')
    print(f'recall: {recall}')
    print(f'precision: {precision}')
    print(f'specificity: {specificity}')
    print(f'accuracy: {accuracy}')
    print(f'f1: {f1}')


def print_document_text(entity_id, dcc_dir):
    """
    Print a document from the DCC dataset based on entity ID
    """
    entity_id_split = entity_id.split('_')
    document_name = entity_id_split[0]
    start = int(entity_id_split[1])
    end = int(entity_id_split[2])
    document_type = document_name[0:2]

    text_path = dcc_dir / document_type / f'{document_name}.txt'
    with open(text_path, 'r') as text_file:
        text = text_file.read()
    print(text)
    print(f'Entity: {text[start: end]} ({start}-{end})')
