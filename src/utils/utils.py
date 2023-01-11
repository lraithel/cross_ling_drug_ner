"""Utilities for data pre-processing."""

import json

# from pynvml import *

from itertools import chain

DELIMITER = "@"


with open("src/utils/lang_dict.json", "r") as read_handle:
    lang2id = json.load(read_handle)


# def print_gpu_utilization():
#     nvmlInit()
#     handle = nvmlDeviceGetHandleByIndex(0)
#     info = nvmlDeviceGetMemoryInfo(handle)
#     print(f"GPU memory occupied: {info.used//1024**2} MB.")


def sort_by_language(predictions, labels, languages):
    """Make one set of predictions-labels for each language."""
    by_language = {lang: {"predictions": [], "true_labels": []} for lang in languages}

    for pred, label, language in zip(predictions, labels, languages):
        by_language[language]["predictions"].append(pred)
        by_language[language]["true_labels"].append(label)

    return by_language


def _count_number_of_entities(doc):
    """Count the number of found non-O tags and annotated spans."""
    print("Counting number of NER tags.")
    ner_tags = doc["ner_tags"]
    tag_num = sum(1 for i in ner_tags if i != 0)

    strings_split = []

    # since some of the mentions are MWEs, split them by whitespace to get
    # a more correct count (this won't catch all of the MWEs)
    for x in doc["spans"]["text"]:
        s = x.split()
        for y in s:
            strings_split.append(y)

    return tag_num, len(strings_split)


def calculate_number_of_found_tags(datasets):
    """..."""
    from utils.utils import _count_number_of_entities

    tags_overall = 0
    spans_overall = 0
    for split, data in datasets.items():
        for doc in data:
            num_tags, num_spans = _count_number_of_entities(doc)

            tags_overall += num_tags
            spans_overall += num_spans

    print(
        f"Number of found tags: {tags_overall}\nNumber of annotated spans: {spans_overall}"
    )


def map_languages(documents):
    """Map the language identifiers to IDs."""
    languages = documents["language"]

    mapped_langs = [lang2id[lang] for lang in languages]

    return {"language": mapped_langs}


def chunk_documents_into_windows(documents, window_size=500):
    """..."""
    file_names = documents["file_name"]
    texts = documents["text"]
    languages = documents["language"]
    labels = documents["labels"]

    chunked_texts = []
    chunked_file_names = []
    chunked_languages = []
    chunked_labels = []

    for file_name, text, language, label_list in zip(
        file_names, texts, languages, labels
    ):
        text_chunks = [
            text[i : i + window_size] for i in range(0, len(text), window_size)
        ]
        label_chunks = [
            label_list[i : i + window_size]
            for i in range(0, len(label_list), window_size)
        ]
        file_name_chunks = [
            f"{file_name}{DELIMITER}{j}" for j in range(0, len(text_chunks))
        ]
        language_chunks = [language] * len(text_chunks)

        assert len(text_chunks[0]) == len(
            label_chunks[0]
        ), f"lengths do not match: text_chunks: {len(text_chunks[0])} vs. label_chunks: {len(label_chunks[0])}"

        chunked_texts.extend(text_chunks)
        chunked_file_names.extend(file_name_chunks)
        chunked_languages.extend(language_chunks)
        chunked_labels.extend(label_chunks)

    return {
        "texts": chunked_texts,
        "file_ids": chunked_file_names,
        "labels": chunked_labels,
        "languages": chunked_languages,
    }


def re_combine_windows(dataset, predictions):
    """Recombine the predictions on the chunked documents."""
    characters_per_chunk = dataset["texts"]

    file_ids_per_chunk = dataset["file_ids"]
    chars_per_file = []
    tags_per_file = []
    txt_file_ids = []
    all_files = []
    all_tags = []
    previous_file_name = ""

    for char_chunk, pred_chunk, file_id in zip(
        characters_per_chunk, predictions, file_ids_per_chunk
    ):
        # print(file_id)
        file_name = file_id.split(DELIMITER)[0]

        if file_name != previous_file_name and previous_file_name != "":
            # print(
            #     f"chars_per_file:\n{chars_per_file}\n\ntags per file:\n{tags_per_file}\n\n######################\n"
            # )
            all_files.append("".join(chars_per_file))
            all_tags.append(tags_per_file)
            txt_file_ids.append(previous_file_name)
            # empty the lists and add the newly collected
            chars_per_file = [char_chunk]
            tags_per_file = pred_chunk

        else:
            chars_per_file.extend(char_chunk)
            tags_per_file.extend(pred_chunk)

        previous_file_name = file_name

    # print(all_tags, all_files, txt_file_ids)

    all_files.append("".join(chars_per_file))
    all_tags.append(tags_per_file)
    txt_file_ids.append(previous_file_name)

    assert len(all_tags) == len(
        all_files
    ), f"#text and #predictions does not match: texts: {len(all_files)} vs. predictions: {len(all_tags)}"
    # return a list of results per document
    return all_tags, all_files, txt_file_ids


def chunk_documents(documents, num_sentences=3, unify_tags=False):
    """Split each document in a sequence of `num_sentences` sentences."""

    file_name = documents["file_name"]

    chunks_tokens = []
    chunks_tags = []
    file_ids = []
    languages = []

    for file_name, row, lang in zip(
        documents["file_name"], documents["tokens_per_sentence"], documents["language"]
    ):
        chunks = [row[i : i + num_sentences] for i in range(0, len(row), num_sentences)]
        chunks_tokens += chunks

        # add file IDs for every chunk (original file base name + )
        file_ids += [f"{file_name}{DELIMITER}{j}" for j in range(0, len(chunks))]
        # add the language for every chunk
        languages += [lang2id[lang]] * len(chunks)

    if unify_tags:
        for row in documents["tags_per_sentence"]:
            chunks_tags += [
                row[i : i + num_sentences] for i in range(0, len(row), num_sentences)
            ]

    else:
        for row in documents["token_labels_per_sentence"]:
            chunks_tags += [
                row[i : i + num_sentences] for i in range(0, len(row), num_sentences)
            ]

    # print(chunks_tokens)
    return {
        "chunks_tokens": chunks_tokens,
        "chunks_tags": chunks_tags,
        "file_ids": file_ids,
        "language": languages,
    }


def combine_x_sentences(documents):
    """Combine the lists containing x sentences to one list of tokens."""
    chunks_tokens_resolved = []
    chunks_tags_resolved = []

    for row in documents["chunks_tokens"]:
        chunks_tokens_resolved.append(list(chain(*row)))

    for row in documents["chunks_tags"]:
        chunks_tags_resolved.append(list(chain(*row)))

    # print(chunks_tokens_resolved)

    return {
        "chunks_tokens": chunks_tokens_resolved,
        "chunks_tags": chunks_tags_resolved,
    }


def resolve_sentences(documents):
    """Combine the lists containing x sentences to one list of tokens."""
    sentences_resolved = []
    tags_resolved = []

    for row in documents["tokens_per_sentence"]:
        sentences_resolved.append(list(chain(*row)))

    for row in documents["token_labels_per_sentence"]:
        tags_resolved.append(list(chain(*row)))

    return {
        "tokens_per_sentence": sentences_resolved,
        "token_labels_per_sentence": tags_resolved,
    }


def re_combine_documents(dataset, predictions):
    """Re-construct the documents from the chunks and map the corresponding predictions."""
    # get the sub-tokens per chunk (e.g. one chunk corresponds to 3 sentences, original
    # tokens might correspond to more than one subtoken), and the file ids
    subtokens_per_chunk = dataset["sub_tokens"]

    file_ids_per_chunk = dataset["file_ids"]

    # ["<s>", "▁pri", "mär", "▁guter", "▁Verlauf", "▁nach", "▁", "TX"]
    # ['B-Drug', 'B-Drug', 'B-Drug', 'B-Drug', 'B-Drug', 'B-Drug', 'B-Drug']
    # create a list of recombined tokens (out of sub-tokens) and the
    # corresponding tags
    new_tokens = []
    tags = []
    for subtokens, preds in zip(subtokens_per_chunk, predictions):
        re_combined_tokens = []
        re_combined_tags = []
        previous_token = ""
        previous_tag = ""

        # add dummy entries for the first and last token (makes iteration easier)
        preds = [-100] + preds + [-100]

        # iterate over sub-tokens and predictions and combine sub-tokens to
        # tokens and "sub"-predictions to tags
        for i, (sub_token, tag) in enumerate(zip(subtokens, preds)):

            # ignore the <s> token
            if i == 0:
                continue
            if sub_token in ("[PAD]", "[SEP]", "<pad>", "<sep>"):
                re_combined_tokens.append(previous_token)
                re_combined_tags.append(previous_tag)

                break  # changed for drug ner

            if sub_token.startswith("▁"):
                if previous_token != "":
                    re_combined_tokens.append(previous_token)
                    re_combined_tags.append(previous_tag)
                # combine sub-tokens
                previous_token = sub_token.replace("▁", "")
                # always take the tag of the first part of the word (not so sure about that)
                previous_tag = tag

            elif sub_token == "</s>" and previous_token != "":

                re_combined_tokens.append(previous_token)
                re_combined_tags.append(previous_tag)

            elif not sub_token.startswith("▁") and sub_token != "</s>":
                previous_token += sub_token

        new_tokens.append(re_combined_tokens)
        tags.append(re_combined_tags)

    # predictions is a list of predictions per chunk,
    # i.e. tokens per chunk and predictions should be always the same size
    tokens_per_file = []
    tags_per_file = []
    txt_files = []
    all_files = []
    all_tags = []
    previous_file_name = ""

    for tokens, re_combined_tags, file_id in zip(new_tokens, tags, file_ids_per_chunk):
        # print(file_id)
        file_name = file_id.split(DELIMITER)[0]

        if file_name != previous_file_name and previous_file_name != "":
            all_files.append(tokens_per_file)
            all_tags.append(tags_per_file)
            txt_files.append(previous_file_name)
            # empty the lists and add the newly collected
            tokens_per_file = [tokens]
            tags_per_file = [re_combined_tags]

        else:
            tokens_per_file.append(tokens)
            tags_per_file.append(re_combined_tags)

        previous_file_name = file_name

    all_files.append(tokens_per_file)
    all_tags.append(tags_per_file)
    txt_files.append(previous_file_name)

    # return a list of results per document
    return all_tags, all_files, txt_files


def re_combine_longformer_tokens(dataset, predictions):
    """Re-construct the documents from the chunks and map the corresponding predictions."""
    # get the sub-tokens per chunk (e.g. one chunk corresponds to 3 sentences, original
    # tokens might correspond to more than one subtoken), and the file ids
    subtokens_per_chunk = dataset["sub_tokens"]
    file_ids_per_chunk = dataset["file_ids"]

    new_tokens = []
    tags = []

    # the following recombines the subtokens to tokens and aligns them with the
    # predicted tags
    for subtokens, preds in zip(subtokens_per_chunk, predictions):
        re_combined_tokens = []
        re_combined_tags = []
        previous_token = ""
        previous_tag = ""

        # add dummy entries for the first and last token (makes iteration easier)
        preds = [-100] + preds + [-100]

        # assert len(subtokens) == len(preds), f"\n{preds}\n\n{subtokens}"
        # iterate over sub-tokens and predictions and combine sub-tokens to
        # tokens and "sub"-predictions to tags
        for i, (sub_token, tag) in enumerate(zip(subtokens, preds)):
            # print(f"\nsubtoken: {sub_token}, tag: {tag}")
            # ignore the [SEP]/<s> token
            if i == 0:
                continue
            if sub_token in ("[PAD]", "<pad>"):
                # print("ignoring special token")
                continue
            if not sub_token.startswith("Ġ"):  # and sub_token != "<s>":

                # if we reached the sentence end marker
                if sub_token == "</s>":
                    re_combined_tokens.append(previous_token.replace("Ġ", ""))
                    re_combined_tags.append(previous_tag)

                    continue
                else:
                    # combine sub-tokens
                    previous_token += sub_token
                    # print(f"previous_token: {previous_token}")

            else:
                # print(f"prev: {previous_token}, prev tag: {previous_tag}, sub_token: {sub_token}, tag: {tag}")
                if previous_token != "":
                    re_combined_tokens.append(previous_token.replace("Ġ", ""))
                    re_combined_tags.append(previous_tag)
                previous_token = sub_token
                previous_tag = tag

        assert len(re_combined_tags) == len(
            re_combined_tokens
        ), f"\n{re_combined_tags}\n\n{re_combined_tokens}"

        new_tokens.append(re_combined_tokens)
        tags.append(re_combined_tags)

    print("compare lengths", len(new_tokens[0]) == len(tags[0]))

    # predictions is a list of predictions per chunk,
    # i.e. tokens per chunk and predictions should be always the same size
    tokens_per_file = []
    tags_per_file = []
    txt_files = []
    all_files = []
    all_tags = []
    previous_file_name = ""

    for tokens, re_combined_tags, file_id in zip(new_tokens, tags, file_ids_per_chunk):

        file_name = file_id.split("_")[0]

        if file_name != previous_file_name and previous_file_name != "":
            all_files.append(tokens_per_file)
            all_tags.append(tags_per_file)
            txt_files.append(previous_file_name)
            # empty the lists and add the newly collected
            tokens_per_file = [tokens]
            tags_per_file = [re_combined_tags]

        else:
            tokens_per_file.append(tokens)
            tags_per_file.append(re_combined_tags)

        previous_file_name = file_name

    all_files.append(tokens_per_file)
    all_tags.append(tags_per_file)
    txt_files.append(previous_file_name)

    # return a list of results per document
    return all_tags, all_files, txt_files
