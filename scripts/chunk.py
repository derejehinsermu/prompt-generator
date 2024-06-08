from tqdm.auto import tqdm
import re

def split_list(input_list: list[str], slice_size: int) -> list[list[str]]:
    return [input_list[i:i + slice_size] for i in range(0, len(input_list), slice_size)]

def chunk_sentences(pages_and_texts: list[dict], num_sentence_chunk_size: int) -> list[dict]:
    for item in tqdm(pages_and_texts):
        item["sentence_chunks"] = split_list(input_list=item["sentences"], slice_size=num_sentence_chunk_size)
        item["num_chunks"] = len(item["sentence_chunks"])
    return pages_and_texts

def create_chunks(pages_and_texts: list[dict]) -> list[dict]:
    pages_and_chunks = []
    for item in tqdm(pages_and_texts):
        for sentence_chunk in item["sentence_chunks"]:
            chunk_dict = {"page_number": item["page_number"]}
            joined_sentence_chunk = "".join(sentence_chunk).replace("  ", " ").strip()
            joined_sentence_chunk = re.sub(r'\.([A-Z])', r'. \1', joined_sentence_chunk)

            chunk_dict["sentence_chunk"] = joined_sentence_chunk
            chunk_dict["chunk_char_count"] = len(joined_sentence_chunk)
            chunk_dict["chunk_word_count"] = len(joined_sentence_chunk.split(" "))
            chunk_dict["chunk_token_count"] = len(joined_sentence_chunk) / 4

            pages_and_chunks.append(chunk_dict)
    return pages_and_chunks
