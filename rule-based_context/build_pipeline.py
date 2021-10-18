#!/usr/bin/env python
# coding: utf-8

import pathlib
import spacy
from spacy.tokens import Span, Doc
from medspacy.context import ConTextComponent

def build_pipeline(path_to_rules):

    # Initialize spacy pipeline

    nlp = spacy.load("nl_core_news_sm",
                     disable=["tagger", "ner"])  # only need the tokenizer, and the parser for sentence splitting
    Span.set_extension("entity_id", default=None)
    Span.set_extension("negation", default=None)
    Doc.set_extension("doc_id", default = None)

    # Modify tokenizer
    # - sometimes sub-words are annotated (e.g. "syndroom" in "ME-syndroom")
    # - text is dirty and contains a lot of punctuation not followed by a space (but by next word)

    infixes = nlp.Defaults.infixes + [r'''[-\\/.+,?]'''] # also split on all of these punctuation marks
    # add to tokenizer
    infix_regex = spacy.util.compile_infix_regex(infixes)
    nlp.tokenizer.infix_finditer = infix_regex.finditer
    nlp.tokenizer.url_match = None  # so text with e.g. a "." between letters doesn't get matched as a url

    # Add labeller component
    # Normally, this would be your NER model or what have you, to recognize the entities that you want to determine the context of (i.e. whether they're negated or not).
    # Since we're working with already labelled data here, we'll instead build a component to add our annotations as entities

    def labeller(doc, doc_id, labels, starts, ends):
        for i, (label, start, end) in enumerate(zip(labels, starts, ends)): 
            new_ent = doc.char_span(start, end, label="CONCEPT", # make annotated entity a spacy span
                                    alignment_mode='expand')  # if [annotation] falls within a token (e.g. "[trauma]'s")
            new_ent._.entity_id = f'{doc_id}_{start}_{end}' # add entity_id as an attribute
            new_ent._.negation = label # add true label as an attribute
            # add this span to list of ents in doc
            orig_ents = list(doc.ents)
            doc.ents = orig_ents + [new_ent]
        doc._.doc_id = doc_id # add doc_id as an attribute
        return doc

    nlp.add_pipe(labeller, first=True)

    # Add Context component

    context = ConTextComponent(nlp, rules="other", rule_list=path_to_rules)

    nlp.add_pipe(context)
    return nlp