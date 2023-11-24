from natasha import (
    Segmenter,
    NewsEmbedding,
    NewsNERTagger,
    Doc
)

from ipymarkup import format_span_box_markup

def get_ner_format(text):
    emb = NewsEmbedding()
    ner_tagger = NewsNERTagger(emb)
    segmenter = Segmenter()

    doc = Doc(text)
    doc.segment(segmenter)
    doc.tag_ner(ner_tagger)
    html = ''.join(list(format_span_box_markup(doc.text, doc.spans)))
    return html