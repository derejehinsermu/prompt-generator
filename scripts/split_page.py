from spacy.lang.en import English
from tqdm.auto import tqdm

def split_text_into_sentences(pages_and_texts: list[dict]) -> list[dict]:
    nlp = English()
    nlp.add_pipe("sentencizer")

    for item in tqdm(pages_and_texts):
        item["sentences"] = list(nlp(item["text"]).sents)
        item["sentences"] = [str(sentence) for sentence in item["sentences"]]
        item["page_sentence_count_spacy"] = len(item["sentences"])

    return pages_and_texts

