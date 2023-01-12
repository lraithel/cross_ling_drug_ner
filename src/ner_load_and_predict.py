import argparse
import json
import numpy as np
import os
import pandas as pd
import re
import torch
import unicodedata


from torch.utils.data import DataLoader

from fine_tune_ner_2 import DrugNER

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments

import evaluate

from bio2brat import convert
from seqeval.metrics import classification_report

from utils import utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

with open("src/utils/lang_dict.json", "r") as read_handle:
    lang2id = json.load(read_handle)
    id2lang = {value: key for key, value in lang2id.items()}


def predict(model, dataset, label_list, reports_file):
    """..."""
    print("Final predictions on test set ...")
    languages = dataset["language"]
    reports = {}

    output = model.predict(dataset)
    predictions = np.argmax(output[0], axis=2)

    labels = output[1]
    # Remove ignored index (special tokens)
    converted_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    # load the traditional and "fair" eval metrics
    trad_eval = evaluate.load("seqeval")
    fair_eval = evaluate.load("hpi-dhc/FairEval", suffix=False, scheme="IOB2")

    cls_report = classification_report(
        y_true=true_labels, y_pred=converted_predictions, output_dict=True
    )
    reports["classification_report"] = cls_report

    sorted_by_language = utils.sort_by_language(
        predictions=converted_predictions, labels=true_labels, languages=languages
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

    return converted_predictions


def load_model_from_checkpoint(path_to_checkpoint, trainer, batch_size):
    """Load the model from the given checkpoint and prepare it for inference."""
    print("\nLoading model weights from checkpoint ...\n")
    # loading the model previously fine-tuned
    print(path_to_checkpoint)
    model = AutoModelForTokenClassification.from_pretrained(path_to_checkpoint)

    # arguments for Trainer
    test_args = TrainingArguments(
        output_dir=path_to_checkpoint,
        do_train=False,
        do_predict=True,
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
    )

    # init trainer
    trainer = Trainer(
        model=model, args=test_args
    )  # , compute_metrics=trainer.compute_metrics
    # )

    return trainer


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

    tups.update(set(matches))

    new_annos = []
    for i, tup in enumerate(tups):
        new_annos.append(f"T{i+1}\tDrug {tup[1]} {tup[2]}\t{tup[0]}")

    return new_annos


def write_conll(output_dir, conll_str, file_name):
    dir_ = "/".join(output_dir.split("/")[:-1])
    conll_anno_dir = os.path.join(dir_, "conll_predictions")
    if not os.path.exists(conll_anno_dir):
        os.makedirs(conll_anno_dir)

    with open(os.path.join(conll_anno_dir, file_name), "w") as conll_handle:

        conll_str = unicodedata.normalize("NFKD", conll_str)
        try:
            conll_handle.write(conll_str)
        except UnicodeEncodeError as e:
            conll_handle.write(conll_str.encode("utf-8"))
        except TypeError as e:
            print(f"'{conll_str}'")
            raise e


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


if __name__ == "__main__":

    checkpoint_dirs = [
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/all_on_all/final_models_all_on_all/netscratch/raithel/projects/cross_ling_drug_detection/models/checkpoint_xlm-roberta-base_22_12_22_17_46",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/all_on_all/final_models_all_on_all/netscratch/raithel/projects/cross_ling_drug_detection/models/checkpoint_xlm-roberta-base_22_12_22_17_50",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/all_on_all/final_models_all_on_all/netscratch/raithel/projects/cross_ling_drug_detection/models/checkpoint_xlm-roberta-base_22_12_22_19_13",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/all_on_all/final_models_all_on_all/netscratch/raithel/projects/cross_ling_drug_detection/models/checkpoint_xlm-roberta-base_22_12_22_19_15",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/all_on_all/final_models_all_on_all/netscratch/raithel/projects/cross_ling_drug_detection/models/checkpoint_xlm-roberta-base_22_12_23_14_40",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_en/mono_en_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_15_33",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_en/mono_en_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_15_52",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_en/mono_en_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_16_10",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_en/mono_en_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_16_24",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_en/mono_en_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_16_42",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_de/mono_de_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_13_56",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_de/mono_de_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_16_10",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_de/mono_de_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_17_47",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_de/mono_de_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_19_12",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_de/mono_de_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_20_59",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_fr/mono_fr_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_12_39",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_f/rmono_fr_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_12_51",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_fr/mono_fr_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_07",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_fr/mono_fr_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_18",
        # "/home/lisa/projects/cross_ling_drug_ner/models_final/mono_ling_fr/mono_fr_models/netscratch/raithel/projects/cross_ling_drug_detection/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_32",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_13_56",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_16_10",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_17_47",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_19_12",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/de/checkpoint_xlm-roberta-base_22_12_24_20_59",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/all_on_all/checkpoint_xlm-roberta-base_22_12_22_17_46",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/all_on_all/checkpoint_xlm-roberta-base_22_12_22_17_50",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/all_on_all/checkpoint_xlm-roberta-base_22_12_22_19_13",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/all_on_all/checkpoint_xlm-roberta-base_22_12_22_19_15",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/all_on_all/checkpoint_xlm-roberta-base_22_12_23_14_40",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_32",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_18",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_07",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_12_51",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_12_39"
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/de_en/checkpoint_xlm-roberta-base_22_12_26_17_54",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/de_en/checkpoint_xlm-roberta-base_22_12_26_17_19",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/de_en/checkpoint_xlm-roberta-base_22_12_26_16_28",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/de_en/checkpoint_xlm-roberta-base_22_12_26_15_42",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/de_en/checkpoint_xlm-roberta-base_22_12_26_21_10"
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/es_fr/checkpoint_xlm-roberta-base_22_12_26_14_31",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/es_fr/checkpoint_xlm-roberta-base_22_12_26_17_34",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/es_fr/checkpoint_xlm-roberta-base_22_12_26_13_47",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/es_fr/checkpoint_xlm-roberta-base_22_12_26_13_06",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/clusters/es_fr/checkpoint_xlm-roberta-base_22_12_26_12_59"
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_32",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_18",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_13_07",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_12_51",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/fr/checkpoint_xlm-roberta-base_22_12_24_12_39"
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_16_42",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_16_24",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_16_10",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_15_52",
        # "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/en/checkpoint_xlm-roberta-base_22_12_24_15_33"
        "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/es/checkpoint_xlm-roberta-base_22_12_24_16_41",
        "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/es/checkpoint_xlm-roberta-base_22_12_24_16_23",
        "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/es/checkpoint_xlm-roberta-base_22_12_24_16_11",
        "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/es/checkpoint_xlm-roberta-base_22_12_24_15_48",
        "/netscratch/raithel/projects/cross_ling_drug_ner/models/by_language/es/checkpoint_xlm-roberta-base_22_12_24_15_31"
 ]

    for no, checkpoint_dir in enumerate(checkpoint_dirs):

        print(f"\nRunning model nr. {no + 1}/{len(checkpoint_dirs)} for inference.\n")

        parser = argparse.ArgumentParser()

        parser.add_argument("config", default=None, help="Path to config file.")

        args = parser.parse_args()

        # get the model information (checkpoint, labels, label2id, etc.)
        # checkpoint_dir = "outputs/checkpoint_xlm-roberta-base_22_12_20_13_00"
        # get the config file of the model
        checkpoint_config_file = os.path.join(checkpoint_dir, "config.json")

        with open(checkpoint_config_file, "r") as read_handle:
            cp_config = json.load(read_handle)

        drug_ner = DrugNER(args.config, mode="eval")

        drug_ner.label_list = list(cp_config["id2label"].values())
        # for some reason, the IDs are strings
        drug_ner.label2id = {
            int(key): value for key, value in cp_config["id2label"].items()
        }
        drug_ner.id2label = cp_config["label2id"]

        # prepare the data (train & dev are prepared as well)
        drug_ner.prepare_data(model=cp_config["_name_or_path"])
        # if we already created a predictions file, we don't have to run everything
        # again
        if not os.path.isfile(os.path.join(checkpoint_dir, "predictions.json")):

            drug_ner.get_model(model=cp_config["_name_or_path"])

            trainer = load_model_from_checkpoint(
                checkpoint_dir, drug_ner, batch_size=drug_ner.config["batch_size"]
            )

            predictions = predict(
                model=trainer,
                dataset=drug_ner.tokenized_datasets["test"],
                label_list=drug_ner.label_list,
                reports_file=os.path.join(checkpoint_dir, "report"),
            )

            with open(os.path.join(checkpoint_dir, "predictions.json"), "w") as d:
                json.dump({"predictions": predictions}, d)

        else:

            print("Opening existing predictions file.")
            with open(os.path.join(checkpoint_dir, "predictions.json"), "r") as d:
                predictions = json.load(d)["predictions"]

        # print(f"predictions:\n{predictions}\n")
        true_labels = [
            [drug_ner.label_list[l] for l in label if l != -100]
            for label in drug_ner.tokenized_datasets["test"]["labels"]
        ]
        # print(f"\nlabels:\n{true_labels}\n")

        print(
            classification_report(
                y_true=true_labels, y_pred=predictions, output_dict=False
            )
        )

        # continue

        # transform the sentence chunks back to sentences per document
        # `predictions` is a list of lists of tags
        combined_predictions, combined_tokens, txt_files = utils.re_combine_documents(
            drug_ner.tokenized_datasets["test"], predictions
        )

        dir_wo_string_matching = "predicted_annotations/"

        output_dir_wosm = os.path.join(checkpoint_dir, dir_wo_string_matching)

        if not os.path.exists(output_dir_wosm):
            os.makedirs(output_dir_wosm)

        convert_documents_to_brat(
            predictions=combined_predictions,
            tokens=combined_tokens,
            txt_files=txt_files,
            data_url=drug_ner.data_url,
            output_dir=output_dir_wosm,
            test_data_identifier=drug_ner.config["test"],
        )
