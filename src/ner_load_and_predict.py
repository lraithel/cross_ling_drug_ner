import argparse
import json
import numpy as np
import os
import re
import torch

from torch.utils.data import DataLoader

from finetune_ner import DrugNER

from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import Trainer
from transformers import TrainingArguments

from bio2brat import convert
from utils import utils

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def predict(model, dataset, label_list):
    """..."""
    print("Final predictions on test set ...")

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


# def load_model_from_checkpoint(path_to_checkpoint, batch_size):
#     """..."""
#     pass


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
        conll_handle.write(conll_str)


def convert_documents_to_brat(
    predictions, tokens, txt_files, data_url, output_dir, drugs=[], drug_length=3
):
    """Convert every given document to a brat file."""
    # convert the predictions per document back to brat documents
    for prediction, tokens, txt_file in zip(predictions, tokens, txt_files):

        assert len(tokens) == len(prediction)

        print(f"Current text file: {txt_file}")
        # path_to_text = os.path.join(data_url, "dev", txt_file + ".txt")
        path_to_text = os.path.join(data_url, "test_test", txt_file + ".txt")

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
                write_handle.write(str(line))
                write_handle.write("\n")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Path to config file.")

    args = parser.parse_args()

    # get the model information (checkpoint, labels, label2id, etc.)
    # checkpoint_dir = "outputs/checkpoint_xlm-roberta-base_22_12_20_13_00"
    checkpoint_dir = "/netscratch/raithel/projects/cross_ling_drug_detection/models/checkpoint_xlm-roberta-base_22_12_22_19_15"
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
        )

        with open(os.path.join(checkpoint_dir, "predictions.json"), "w") as d:
            json.dump({"predictions": predictions}, d)

    else:

        print("Opening existing predictions file.")
        with open(os.path.join(checkpoint_dir, "predictions.json"), "r") as d:
            predictions = json.load(d)["predictions"]

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
        drugs=[],
    )
