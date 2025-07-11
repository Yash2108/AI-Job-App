import spacy
nlp = spacy.load("en_core_web_sm")  # Load once globally

def extract_pos_keywords(text):
    doc = nlp(text)
    keywords = set()

    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"] and not token.is_stop and token.is_alpha:
            keywords.add(token.text.lower())

    return keywords