import argparse
import evaluate
import json
import os
import random
import re

import pandas as pd

from bio2brat import convert
from fine_tune_ner_2 import DrugNER
from ner_load_and_predict import write_conll
from utils import utils

from collections import Counter
from seqeval.metrics import classification_report

with open("src/utils/lang_dict.json", "r") as read_handle:
    lang2id = json.load(read_handle)
    id2lang = {value: key for key, value in lang2id.items()}

TAG_LIST = [
    "NoDisposition",
    "Disposition",
    "Undetermined",
    "Substance",
    "Medication",
    "MEDICATION",
    "substance",  # seems to be not really related to medication
    "CHEM",
    "Drug",
    "NO_NORMALIZABLES",  # make sure NO_NORMALIZABLES is listed before NORMALIZABLES
    "NORMALIZABLES",
]


def calculate_scores(true_labels, merged_predictions, languages, reports_file):
    """Calculate the scores once again, using the merged predictions."""

    reports = {}

    # load the traditional and "fair" eval metrics
    trad_eval = evaluate.load("seqeval")
    fair_eval = evaluate.load("hpi-dhc/FairEval", suffix=False, scheme="IOB2")

    cls_report = classification_report(
        y_true=true_labels, y_pred=merged_predictions, output_dict=True
    )
    reports["classification_report"] = cls_report

    sorted_by_language = utils.sort_by_language(
        predictions=merged_predictions, labels=true_labels, languages=languages
    )

    for language, outputs in sorted_by_language.items():

        results_trad_per_lang = trad_eval.compute(
            predictions=outputs["predictions"],
            references=outputs["true_labels"],
            suffix=False,
        )

        reports[f"{id2lang[language]}_traditional"] = results_trad_per_lang

        try:
            results_fair_per_lang = fair_eval.compute(
                predictions=outputs["predictions"],
                references=outputs["true_labels"],
                mode="fair",
                error_format="count",
            )

            reports[f"{id2lang[language]}_fair"] = results_fair_per_lang

        except ValueError:
            print(f"Warning: could not get results for language '{id2lang[language]}'")
            pass

    for key, results_dict in reports.items():
        print(f"\n{key}")
        try:
            df = pd.DataFrame.from_dict(results_dict)
            df.to_csv(reports_file + f"_{key}.csv")
            print(df)
            print("\n\n")
        except ValueError as e:
            print(e)
            pass


def convert_documents_to_brat(
    predictions, tokens, txt_files, data_url, output_dir, test_data_identifier
):
    """Convert every given document to a brat file."""
    # convert the predictions per document back to brat documents
    print("\nConverting documents to brat ...\n")
    for prediction, tokens, txt_file in zip(predictions, tokens, txt_files):

        assert len(tokens) == len(prediction)

        print(f"Current text file: {txt_file}")
        # path_to_text = os.path.join(data_url, "dev", txt_file + ".txt")
        path_to_text = os.path.join(data_url, test_data_identifier, txt_file + ".txt")

        # returns a list of annotation strings
        brat_anno, bio_str = convert(
            text_file=path_to_text, model_predictions=prediction, tokens=tokens
        )

        file_name = os.path.basename(txt_file).split(".txt")[0]
        write_conll(
            output_dir=output_dir, conll_str=bio_str, file_name=f"{file_name}.conll"
        )

        path = os.path.join(output_dir, f"{file_name}.ann")

        # create one file for every document
        with open(path, "w") as write_handle:
            for line in brat_anno:
                # line = unicodedata.normalize("NFKD", str(line))

                write_handle.write(str(line))
                write_handle.write("\n")


def agree_5(a, b, c, d, e):
    return a == b == c == d == e


def get_tag(freq_list):
    """If there is a tie between majorities, always settle on the I-drug."""

    for tag_variant in TAG_LIST:

        if freq_list[0][0].endswith(tag_variant) and freq_list[1][0].endswith(
            tag_variant
        ):
            return f"I-{tag_variant}"

        if freq_list[0][0].endswith(tag_variant) and not freq_list[1][0].endswith(
            tag_variant
        ):
            return f"I-{tag_variant}"

    # if something tagged as drug has the same amount of votes as the "O",
    # always choose drug (and I-Drug, because it is converted to B if necessary)
    if freq_list[0][0] == "O":
        return "I-Drug"
    # if we have a tie between two different labels (that are not "O"),
    # randomly choose either the first or the second
    elif freq_list[0][0] != "O" and freq_list[0][1] != "O":
        rand_int = random.randint(0, 1)
        return freq_list[rand_int][0]
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

            if agree_5(tag1, tag2, tag3, tag4, tag5):
                # print("tag1 = tag2")
                agreed_tag = tag1
                # print(f"agreed: {tag1} vs. {tag2} --> {agreed_tag}")

            elif not agree_5(tag1, tag2, tag3, tag4, tag5):
                not_agreed_counter += 1

                counter = Counter([tag1, tag2, tag3, tag4, tag5])

                majority_list = counter.most_common()
                # print(f"majority list: {majority_list}")

                if has_tie(majority_list):
                    agreed_tag = get_tag(majority_list)

                else:
                    agreed_tag = majority_list[0][0]

            else:
                print(tag1, tag2, tag3, tag4, tag5)
                assert False

            final.append(agreed_tag)
            previous_tag = agreed_tag

        final_predictions.append(final)
        final = []
        previous_tag = "O"

    print(f"not agreed counter: {not_agreed_counter}/{len(pred_1)}")

    return final_predictions


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Path to config file.")

    parser.add_argument("model_1", default=None, help="model path 1.")
    parser.add_argument("model_2", default=None, help="model path 2.")
    parser.add_argument("model_3", default=None, help="model path 3.")
    parser.add_argument("model_4", default=None, help="model path 4.")
    parser.add_argument("model_5", default=None, help="model path 5.")

    args = parser.parse_args()

    # assert args.model_1 != args.model_2 != args.model_3 != args.model_4 != args.model_5

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
    # merged the predictions by majority vote
    merged_predictions = merge_predictions_5_models(
        predictions_1, predictions_2, predictions_3, predictions_4, predictions_5
    )
    # get the true labels
    true_labels = [
        [drug_ner.label_list[l] for l in label if l != -100]
        for label in drug_ner.tokenized_datasets["test"]["labels"]
    ]
    # get the language for each test set sample
    languages = drug_ner.tokenized_datasets["test"]["language"]

    print(
        classification_report(
            y_true=true_labels, y_pred=merged_predictions, output_dict=False
        )
    )

    target_folder = "ensembled_predictions"

    path = "/".join(args.model_1.split("/")[:-1])
    print(path)
    predictions_output_dir = os.path.join(path, target_folder)

    if not os.path.exists(predictions_output_dir):
        os.makedirs(predictions_output_dir)

    calculate_scores(
        true_labels=true_labels,
        merged_predictions=merged_predictions,
        reports_file=os.path.join(path, "report_ensembled"),
        languages=languages,
    )

    # transform the sentence chunks back to sentences per document
    # `predictions` is a list of lists of tags
    combined_predictions, combined_tokens, txt_files = utils.re_combine_documents(
        drug_ner.tokenized_datasets["test"], merged_predictions
    )

    convert_documents_to_brat(
        predictions=combined_predictions,
        tokens=combined_tokens,
        txt_files=txt_files,
        data_url=drug_ner.data_url,
        output_dir=predictions_output_dir,
        test_data_identifier=drug_ner.config["test"],
    )
