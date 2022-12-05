"""Utilities for data pre-processing."""

from itertools import chain

DELIMITER = "@"


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

    # tokens = doc["tokens"]
    # token_tag = []
    # for token, tag in zip(tokens, ner_tags,):
    #     if tag != 0:
    #         token_tag.append((token, tag))

    # for (token, tag), str_span in zip(token_tag, strings_split):
    #     print(token, tag, str_span)

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


def chunk_documents(documents, num_sentences=3, unify_tags=False):
    """Split each document in a sequence of `num_sentences` sentences."""

    file_name = documents["file_name"]

    chunks_tokens = []
    chunks_tags = []
    file_ids = []

    for file_name, row in zip(documents["file_name"], documents["tokens_per_sentence"]):
        chunks = [row[i : i + num_sentences] for i in range(0, len(row), num_sentences)]
        chunks_tokens += chunks

        file_ids += [f"{file_name}{DELIMITER}{j}" for j in range(0, len(chunks))]

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

    return {
        "chunks_tokens": chunks_tokens,
        "chunks_tags": chunks_tags,
        "file_ids": file_ids,
    }


def combine_x_sentences(documents):
    """Combine the lists containing x sentences to one list of tokens."""
    chunks_tokens_resolved = []
    chunks_tags_resolved = []

    for row in documents["chunks_tokens"]:
        chunks_tokens_resolved.append(list(chain(*row)))

    for row in documents["chunks_tags"]:
        chunks_tags_resolved.append(list(chain(*row)))

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

    # print(f"file ids per chunk: {file_ids_per_chunk}")

    new_tokens = []
    tags = []
    for subtokens, preds in zip(subtokens_per_chunk, predictions):
        re_combined_tokens = []
        re_combined_tags = []
        previous_token = ""
        previous_tag = ""

        # print(f"\nsubtokens: {subtokens},\n\npreds: {preds}")

        # add dummy entries for the first and last token (makes iteration easier)
        preds = [-100] + preds + [-100]

        # iterate over sub-tokens and predictions and combine sub-tokens to
        # tokens and "sub"-predictions to tags
        for i, (sub_token, tag) in enumerate(zip(subtokens, preds)):
            # print(f"\nsubtoken: {sub_token}, tag: {tag}")

            # ignore the [SEP] token
            if i == 0:
                continue
            if sub_token in ("[PAD]", "[SEP]"):
                re_combined_tokens.append(previous_token)
                re_combined_tags.append(previous_tag)
                # print("ignoring special token")
                # continue
                break  # changed for drug ner
            if sub_token.startswith("##"):
                # combine sub-tokens
                previous_token += sub_token.replace("##", "")
                # the previous tag stays the same

            else:
                if previous_token != "":
                    re_combined_tokens.append(previous_token)
                    re_combined_tags.append(previous_tag)
                previous_token = sub_token
                previous_tag = tag

        new_tokens.append(re_combined_tokens)
        tags.append(re_combined_tags)

    # print(f"\nnew tokens: {new_tokens}")
    # print(f"\ntags: {tags}")
    # print("\n")

    # predictions is a list of predictions per chunk,
    # i.e. tokens per chunk and predictions should be always the same size
    tokens_per_file = []
    tags_per_file = []
    txt_files = []
    all_files = []
    all_tags = []
    previous_file_name = ""

    for tokens, re_combined_tags, file_id in zip(new_tokens, tags, file_ids_per_chunk):
        print(file_id)
        file_name = file_id.split(DELIMITER)[0]
        # print(f"tokens: {tokens}\nrecombined tags: {re_combined_tags}\n")
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
    # print(f"\nall tokens: {all_files}")
    all_tags.append(tags_per_file)
    txt_files.append(previous_file_name)

    # print(f"all files: {all_files}")

    # print(f"txt files: {txt_files}")

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

    # print(all_tags[:2])
    # print("\n")
    # print(all_files[:2])

    # return a list of results per document
    return all_tags, all_files, txt_files
