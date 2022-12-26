import argparse
import json
import os

from bio2brat import convert
from finetune_ner import DrugNER
from utils import utils

import re

from collections import Counter


def get_string_matches(annos, path_to_text, drugs, drug_length=3):
    """Add simple string matches based on a list of drugs."""
    tups = set()
    for line in annos:
        parts = str(line).split("\t")
        d = parts[-1]
        rem = parts[1].split()
        tups.add((d, rem[1], rem[2]))

    with open(path_to_text, "r") as read_handle:
        text = read_handle.read()

    matches = []
    # longest drug is 96 characters
    for drug in drugs:

        if len(drug) > drug_length:
            matches.extend(
                [
                    (
                        text[match.start() : match.end()],
                        str(match.start()),
                        str(match.end()),
                    )
                    for match in re.finditer(re.escape(drug.lower()), text.lower())
                ]
            )

    # print(f"tups: {tups}\n")
    # if set(matches) - tups:
    #     print(f"added: {set(matches) - tups}")
    tups.update(set(matches))
    # print(f"merged: {tups}\n-----------------------------------------\n")

    new_annos = []
    for i, tup in enumerate(tups):
        new_annos.append(f"T{i+1}\tDrug {tup[1]} {tup[2]}\t{tup[0]}")

    return new_annos


def convert_documents_to_brat(
    predictions, tokens, txt_files, data_url, output_dir, drugs=[], drug_length=3
):
    """Convert every given document to a brat file."""
    # convert the predictions per document back to brat documents
    for prediction, tokens, txt_file in zip(predictions, tokens, txt_files):

        assert len(tokens) == len(prediction)
        path_to_text = os.path.join(data_url, "release1", txt_file + ".txt")
        # returns a list of annotation strings
        brat_anno, bio_str = convert(
            text_file=path_to_text, model_predictions=prediction, tokens=tokens
        )
        if drugs:
            brat_anno = get_string_matches(
                annos=brat_anno,
                path_to_text=path_to_text,
                drugs=drugs,
                drug_length=drug_length,
            )

        file_name = os.path.basename(txt_file).split(".txt")[0]
        path = os.path.join(output_dir, f"{file_name}.ann")

        # create one file for every document
        with open(path, "w") as write_handle:
            for line in brat_anno:
                write_handle.write(str(line))
                write_handle.write("\n")


# def agree_3(a, b, c):
#     return a == b == c


def agree_2(a, b):
    return a == b


def agree_5(a, b, c, d, e):
    return a == b == c == d == e


def merge_predictions(pred_1, pred_2):
    """..."""
    final_predictions = []
    not_agreed_counter = 0
    for p1, p2 in zip(pred_1, pred_2):
        previous_tag = "O"
        final = []

        for tag1, tag2 in zip(p1, p2):
            # print(tag1, tag2)
            # if agree_3(tag1, tag2):
            #     # print("all same")
            #     final.append(tag1)

            if agree_2(tag1, tag2):
                # print("tag1 = tag2")
                agreed_tag = tag1
                print(f"agreed: {tag1} vs. {tag2} --> {agreed_tag}")

            elif not agree_2(tag1, tag2):
                not_agreed_counter += 1

                if tag1 == "O" and tag2.endswith("Drug"):
                    agreed_tag = tag2
                elif tag2 == "O" and tag1.endswith("Drug"):
                    agreed_tag = tag1

                elif tag1.startswith("I") and tag2.startswith("B"):
                    agreed_tag = tag1

                elif tag1.startswith("B") and tag2.startswith("I"):
                    agreed_tag = tag2

                else:
                    assert False

                print(f"did not agree: {tag1} vs. {tag2} --> {agreed_tag}")

            # elif agree_2(tag2):
            #     # print("tag2 = tag3")

            #     final.append(tag2)

            # elif tag1 != tag2 and tag1 != tag3:
            #     # print("tag1 != tag2 != tag3")

            #     final.append(tag1)

            else:
                print(tag1, tag2)
                assert False
            final.append(agreed_tag)
            previous_tag = agreed_tag

        final_predictions.append(final)
        final = []
        previous_tag = "O"

    print(f"not agreed counter: {not_agreed_counter}")

    return final_predictions


def get_tag(freq_list):
    """If there is a tie between majorities, always settle on the I-drug."""

    if freq_list[0][0].endswith("Drug") and freq_list[1][0].endswith("Drug"):
        return "I-Drug"

    if freq_list[0][0].endswith("Drug") and not freq_list[1][0].endswith("Drug"):
        return "I-Drug"

    # if something tagged as drug has the same amount of votes as the "O",
    # always choose drug (and I-Drug, because it is converted to B if necessary)
    if freq_list[0][0] == "O":
        return "I-Drug"
    else:
        assert False, f"can't decide tie: {freq_list}"


def has_tie(freq_list):
    """If there is a tie between majorities, always settle on the I-drug."""

    return freq_list[0][1] == freq_list[1][1]


def merge_predictions_5_models(pred_1, pred_2, pred_3, pred_4, pred_5):
    """..."""
    final_predictions = []
    not_agreed_counter = 0

    for p1, p2, p3, p4, p5 in zip(pred_1, pred_2, pred_3, pred_4, pred_5):
        previous_tag = "O"
        final = []

        for tag1, tag2, tag3, tag4, tag5 in zip(p1, p2, p3, p4, p5):
            # print(tag1, tag2)
            # if agree_3(tag1, tag2):
            #     # print("all same")
            #     final.append(tag1)

            if agree_5(tag1, tag2, tag3, tag4, tag5):
                # print("tag1 = tag2")
                agreed_tag = tag1
                print(f"agreed: {tag1} vs. {tag2} --> {agreed_tag}")

            elif not agree_5(tag1, tag2, tag3, tag4, tag5):
                not_agreed_counter += 1

                counter = Counter([tag1, tag2, tag3, tag4, tag5])

                majority_list = counter.most_common()
                print(f"majority list: {majority_list}")

                if has_tie(majority_list):
                    agreed_tag = get_tag(majority_list)

                else:
                    agreed_tag = majority_list[0][0]

                # if tag1 == "O" and tag2.endswith("Drug"):
                #     agreed_tag = tag2
                # elif tag2 == "O" and tag1.endswith("Drug"):
                #     agreed_tag = tag1

                # elif tag1.startswith("I") and tag2.startswith("B"):
                #     agreed_tag = tag1

                # elif tag1.startswith("B") and tag2.startswith("I"):
                #     agreed_tag = tag2

                # else:
                #     assert False

                print(
                    f"did not agree: {tag1} vs. {tag2} vs. {tag3}  vs. {tag4} vs. {tag5} --> {agreed_tag}"
                )

            # elif agree_2(tag2):
            #     # print("tag2 = tag3")

            #     final.append(tag2)

            # elif tag1 != tag2 and tag1 != tag3:
            #     # print("tag1 != tag2 != tag3")

            #     final.append(tag1)

            else:
                print(tag1, tag2, tag3, tag4, tag5)
                assert False

            final.append(agreed_tag)
            previous_tag = agreed_tag

        final_predictions.append(final)
        final = []
        previous_tag = "O"

    print(f"not agreed counter: {not_agreed_counter}")

    return final_predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Path to config file.")

    parser.add_argument("model_1", default=None, help="model path 1.")
    parser.add_argument("model_2", default=None, help="model path 2.")
    parser.add_argument("model_3", default=None, help="model path 3.")
    parser.add_argument("model_4", default=None, help="model path 4.")
    parser.add_argument("model_5", default=None, help="model path 5.")

    # parser.add_argument("model_3", default=None, help="model path 3.")

    parser.add_argument("drug_length", default=3, nargs="+", type=int)

    args = parser.parse_args()

    with open(os.path.join(args.model_1, "predictions.json"), "r") as d:
        predictions_1 = json.load(d)["predictions"]

    with open(os.path.join(args.model_2, "predictions.json"), "r") as d:
        predictions_2 = json.load(d)["predictions"]

    with open(os.path.join(args.model_3, "predictions.json"), "r") as d:
        predictions_3 = json.load(d)["predictions"]

    with open(os.path.join(args.model_4, "predictions.json"), "r") as d:
        predictions_4 = json.load(d)["predictions"]

    with open(os.path.join(args.model_5, "predictions.json"), "r") as d:
        predictions_5 = json.load(d)["predictions"]

    # with open(os.path.join(args.model_3, "predictions.json"), "r") as d:
    #     predictions_3 = json.load(d)["predictions"]

    drug_ner = DrugNER(args.config)

    checkpoint_config_file = os.path.join(args.model_1, "config.json")

    with open(checkpoint_config_file, "r") as read_handle:
        cp_config = json.load(read_handle)

    drug_ner = DrugNER(args.config)

    drug_ner.label_list = list(cp_config["id2label"].values())
    # for some reason, the IDs are strings
    drug_ner.label2id = {
        int(key): value for key, value in cp_config["id2label"].items()
    }
    drug_ner.id2label = cp_config["label2id"]

    # prepare the data (train & dev are prepared as well)
    drug_ner.prepare_data(model=cp_config["_name_or_path"])

    merged_predictions = merge_predictions_5_models(
        predictions_1, predictions_2, predictions_3, predictions_4, predictions_5
    )

    # transform the sentence chunks back to sentences per document
    # `predictions` is a list of lists of tags
    combined_predictions, combined_tokens, txt_files = utils.re_combine_documents(
        drug_ner.tokenized_datasets["test"], merged_predictions
    )

    dir_wo_string_matching = "ensemble_all_on_all/"

    output_dir_wosm = os.path.join("models/final/", dir_wo_string_matching)

    if not os.path.exists(output_dir_wosm):
        os.makedirs(output_dir_wosm)

    convert_documents_to_brat(
        predictions=combined_predictions,
        tokens=combined_tokens,
        txt_files=txt_files,
        data_url=drug_ner.data_url,
        output_dir=output_dir_wosm,
        drugs=[],
    )
