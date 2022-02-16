import pandas as pd
from IPython.core.display import display, HTML


def print_statistics(results, method):
    """
    Calculate statistics
    """
    tp = results[(results.label == 'negated') & (results[method] == 'negated')].shape[0]
    tn = results[(results.label == 'not negated') & (results[method] == 'not negated')].shape[0]
    fp = results[(results.label == 'not negated') & (results[method] == 'negated')].shape[0]
    fn = results[(results.label == 'negated') & (results[method] == 'not negated')].shape[0]
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

def get_document_text(entity_id, dcc_dir, predictions=None, print_text=True, print_html=True, obfuscate_entity=False):
    """
    Print and return a document from the DCC dataset based on entity ID
    """
    entity_id_split = entity_id.split('_')
    document_name = entity_id_split[0]
    start = int(entity_id_split[1])
    end = int(entity_id_split[2])
    document_type = document_name[0:2]

    # Print text
    text_path = dcc_dir / document_type / f'{document_name}.txt'
    with open(text_path, 'r') as text_file:
        text = text_file.read()
        
    def pretty_print(txt, start, end):
        """
        Print a string in html, with part of it highlighted
        """
        snippet = txt[start:end]
    
        def highlight(snippet):
            blob = f"<text>{snippet}</text>"
            blob = f"<mark style='background-color: #fff59d'>{blob}</mark>"
            return blob
    
        display(HTML((''.join((txt[:start], highlight(snippet), txt[end:], '<br>')))))

    if print_text:
        if print_html:
            pretty_print(text, start, end)
        else:
            print(text)
        print(f'Entity: {text[start: end]} ({start}-{end})\n')
        # Print result 
        if predictions is not None:
            print(predictions[predictions.entity_id == entity_id])
            
    if obfuscate_entity:
        # replace the entity with '[ENT]'
        text = text[:start] + '[ENT]' + text[end:]
        
    # Also return text, start and stop for downstream analysis
    return text, start, end
