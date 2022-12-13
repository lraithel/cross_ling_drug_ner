"""..."""
import evaluate

from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_metric
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import CanineForTokenClassification
from transformers import CanineTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import EarlyStoppingCallback
from transformers import Trainer
from transformers import TrainingArguments
from transformers import set_seed


import argparse
import json
import numpy as np
import os
import wandb

from bio2brat import convert
from datetime import datetime
from seqeval.metrics import classification_report
from seqeval.scheme import BILOU
from seqeval.scheme import IOB2
from utils import utils

os.environ["HF_DATASETS_OFFLINE"] = "1"


class DrugNER(object):
    """docstring for DrugNER"""

    def __init__(self, config):
        super(DrugNER, self).__init__()

        # load the "static" config params
        with open(config, "r") as read_handle:
            params = json.load(read_handle)

        self.debug = params["debug"]

        if params["use_wandb"]:
            wandb.init(config=params, project="cross_ling_drug_ner", entity="lraithel")
            self.config = wandb.config
            self.debug = False
        else:
            wandb.init(mode="disabled")
            self.config = params

        print(f"config: {self.config}")

        # set_seed(self.config["seed"])
        self.time = datetime.now().strftime("%d_%m_%y_%H_%M")

        self.data_url = self.config["data_url"]
        print(self.data_url)
        self.out_dir = self.config["out_dir"]

        # will be filled when preparing data
        self.label_list = []
        self.label2id = {}
        self.id2label = {}

        self.metric = evaluate.load("seqeval")
        self.fair_eval = evaluate.load("hpi-dhc/FairEval", suffix=False)

    def get_tokenizer(self, model=False):
        print(model)
        """Load a pre-trained tokenizer."""
        if not model:
            model = self.config["model_name"]

        if self.config["model_name"].startswith("google/canine"):
            self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, use_fast=True, add_prefix_space=True
            )

    def get_model(self, model=None):
        """Load a pre-trained model."""
        if model is None:
            model = self.config["model_name"]

        print(f"\nMODEL: {model}")

        if self.config["model_name"].startswith("google/canine"):
            self.model = CanineForTokenClassification.from_pretrained(
                self.config["model_name"]
            )

        else:
            self.model = AutoModelForTokenClassification.from_pretrained(
                model,
                num_labels=len(self.label_list),
                ignore_mismatched_sizes=True,
                id2label=self.id2label,
                label2id=self.label2id,
            )

    # def convert_to_unicode_id(self, documents):
    #     """Turn each character into its unicode code point id."""

    #     input_ids = torch.tensor([[ord(char) for char in text]])
    #     tokenized_inputs["labels"] = labels
    #     tokenized_inputs["sub_tokens"] = sub_tokens_all

    #     return converted_inputs

    def tokenize_and_align_labels(self, documents, label_all_tokens=True):
        """Align the labels with the IDs.

        The data is already tokenized, but BERT needs subword tokens.
        """
        chunked_tokens = documents["chunks_tokens"]
        chunked_labels = documents["chunks_tags"]

        if self.config["model_name"].startswith("google/canine"):
            tokenized_inputs = {
                "input_ids": [],
                "token_type_ids": [],
                "attention_mask": [],
            }
            for chunk in chunked_tokens:
                inputs = self.tokenizer(
                    chunk, padding="longest", truncation=True, return_tensors="pt"
                )

                assert (
                    len(inputs["input_ids"])
                    == len(inputs["token_type_ids"])
                    == len(inputs["attention_mask"])
                )

                tokenized_inputs["input_ids"].append(inputs["input_ids"])
                tokenized_inputs["token_type_ids"].append(inputs["token_type_ids"])
                tokenized_inputs["attention_mask"].append(inputs["attention_mask"])

            # print(f"\ntokenized_inputs: {tokenized_inputs}")

            return tokenized_inputs
        else:

            # sub-tokenize input sentences and convert to IDs
            tokenized_inputs = self.tokenizer(
                chunked_tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=510,
                padding="max_length",
                # add_prefix_space=True
            )
            # add the original tokens to the dataset
            tokenized_inputs["chunks_tokens"] = documents["chunks_tokens"]

            # we do no need this, only out of curiosity for what the sub word tokens
            # look like
            sub_tokens_all = []
            for idx_list in tokenized_inputs["input_ids"]:
                # input_ids = tokenized_inputs["input_ids"][i]
                sub_tokens = self.tokenizer.convert_ids_to_tokens(idx_list)
                sub_tokens_all.append(sub_tokens)

            labels = []
            # for i, label in enumerate(documents["ner_tags"]):
            for i, chunked_label_list in enumerate(chunked_labels):
                word_ids = tokenized_inputs.word_ids(batch_index=i)

                previous_word_idx = None
                label_ids = []
                for word_idx in word_ids:
                    # Special tokens have a word id that is None. We set the label to -100 so they are automatically
                    # ignored in the loss function.
                    if word_idx is None:
                        label_ids.append(-100)
                    # We set the label for the first token of each word.
                    elif word_idx != previous_word_idx:
                        label_ids.append(self.label2id[chunked_label_list[word_idx]])
                    # For the other tokens in a word, we set the label to either the current label or -100, depending on
                    # the label_all_tokens flag.
                    else:
                        label_ids.append(
                            self.label2id[chunked_label_list[word_idx]]
                            if label_all_tokens
                            else -100
                        )
                    previous_word_idx = word_idx

                labels.append(label_ids)

            assert len(labels) == len(
                sub_tokens_all
            ), "#labels and #sub tokens do not match"

            tokenized_inputs["labels"] = labels
            tokenized_inputs["sub_tokens"] = sub_tokens_all

            return tokenized_inputs

    def compute_metrics(self, p):
        """..."""
        print("Computing metrics ... ")
        predictions, labels = p
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        cleaned_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results_trad = self.metric.compute(
            predictions=cleaned_predictions, references=true_labels, suffix=False
        )
        results_fair = self.fair_eval.compute(
            predictions=cleaned_predictions,
            references=true_labels,
            mode="fair",
            error_format="count",
        )

        print(json.dumps(results_fair, indent=2))

        wandb.log(results_trad)
        wandb.log(results_fair)

        return results_trad

    def predict(self, model):
        """..."""
        print("Final predictions on test set:")

        predictions, labels, _ = model.predict(self.tokenized_datasets["test"])
        predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        converted_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        results = self.metric.compute(
            predictions=converted_predictions, references=true_labels, suffix=False
        )
        wandb.log(results)
        wandb.log({"test_results": results})

        return self.tokenized_datasets["test"], converted_predictions, results

    def train_and_evaluate_model(self, model=None):
        """..."""
        self.get_model(model=model)

        model = self.config["model_name"]
        model_name = model.split("/")[-1]

        if self.debug:
            epochs = 10
            eval_steps = 20  # 500
            save_steps = 20

        else:
            epochs = self.config["epochs"]
            eval_steps = 500  # 500
            save_steps = 500

        out_dir = os.path.join(
            self.out_dir,
            f"{model_name}-finetuned-ner_unifytags_{self.config['unify_tags']}_{self.time}",
        )

        args = TrainingArguments(
            output_dir=out_dir,
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps,  # 500
            save_steps=save_steps,
            learning_rate=self.config["learning_rate"],
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            num_train_epochs=epochs,
            weight_decay=0.01,
            push_to_hub=False,
            seed=self.config["seed"],
            load_best_model_at_end=True,
            metric_for_best_model="overall_f1",
            greater_is_better=True,
            save_total_limit=1,
            report_to="wandb",
        )

        # batches processed examples together while applying padding to make
        # them all the same size
        # if self.config["model_name"] != "google/canine-c":
        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        trainer = Trainer(
            self.model,
            args=args,
            train_dataset=self.tokenized_datasets["train"],
            eval_dataset=self.tokenized_datasets["dev"],
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            compute_metrics=self.compute_metrics,
            callbacks=[
                EarlyStoppingCallback(early_stopping_patience=self.config["patience"])
            ],
        )
        # else:
        # trainer = Trainer(
        #     self.model,
        #     args=args,
        #     train_dataset=self.datasets["train"],
        #     eval_dataset=self.datasets["dev"],
        #     # data_collator=data_collator,
        #     # tokenizer=self.tokenizer,
        #     compute_metrics=self.compute_metrics,
        #     callbacks=[
        #         EarlyStoppingCallback(
        #             early_stopping_patience=self.config["patience"]
        #         )
        #     ],
        # )
        trainer.train()
        trainer.evaluate()

        return trainer

    def prepare_data(self, model=False, download_mode="force_redownload"):
        """..."""
        path_to_loader = "src/brat_dataset.py"

        datasets = load_dataset(
            path_to_loader,
            subdirectory_mapping={
                self.config["train"]: "train",
                self.config["dev"]: "dev",
                self.config["test"]: "test",
            },
            url=self.config["data_url"],
            unify_tags=self.config["unify_tags"],
            remove_all_except_drug=self.config["remove_all_except_drug"],
            # cache_dir="../.cache/huggingface/datasets",
            download_mode="force_redownload",
        )

        # dataset = load_dataset('dfki-nlp/brat', **kwargs)

        print(json.dumps(datasets["train"][0], indent=2))

        if self.config["unify_tags"]:
            self.label_list = datasets["train"].features["ner_tags"].feature.names
        else:
            self.label_list = datasets["train"].features["token_labels"].feature.names

        print(self.label_list)
        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}

        print(json.dumps(self.label2id, indent=2))

        print(
            f"train: {len(datasets['train']['tags_per_sentence'])}, max len: {max([len(sent) for sent in datasets['train']['tags_per_sentence']])}\n"
            f"dev: {len(datasets['dev']['tags_per_sentence'])}, lens {max([len(sent) for sent in datasets['dev']['tags_per_sentence']])}\n"
            f"test: {len(datasets['test']['tags_per_sentence'])}, lens {max([len(sent) for sent in datasets['test']['tags_per_sentence']])}\n"
            f"all: {len(datasets['train']['tags_per_sentence']) + len(datasets['dev']['tags_per_sentence']) + len(datasets['test']['tags_per_sentence'])}\n"
            f""
        )

        # split the documents in chunks of x sentences (results in lists of lists of tokens)
        datasets = datasets.map(
            utils.chunk_documents,
            fn_kwargs={
                "num_sentences": self.config["num_sentences"],
                "unify_tags": self.config["unify_tags"],
            },
            batched=True,
            remove_columns=datasets["train"].column_names,
        )

        # print(f"chunk tokens: {datasets['test']['chunks_tokens']}\n")

        # combine the lists of lists of tokens to lists of tokens (should be done in mapping above)
        self.datasets = datasets.map(
            utils.combine_x_sentences,
            batched=True,
            # those two columns are not really removed but replaced by the
            # same column names
            remove_columns=["chunks_tokens", "chunks_tags"],
        )

        print(self.datasets)

        # if self.config["model_name"] == "google/canine-c":

        #     self.tokenized_datasets = self.datasets.map(
        #         self.convert_to_unicode_id, batched=True
        #     )

        # else:

        self.get_tokenizer(model=model)

        self.tokenized_datasets = self.datasets.map(
            self.tokenize_and_align_labels,
            batched=True,
            fn_kwargs={"label_all_tokens": True},
        )
        # print(json.dumps(self.tokenized_datasets["train"][0], indent=2))

    def train_model(self, model=None):
        trained_model = self.train_and_evaluate_model(model=model)
        return trained_model

    def evaluate_on_testset(self, model):
        return self.predict(model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("config", default=None, help="Path to config file.")

    args = parser.parse_args()

    drug_ner = DrugNER(args.config)

    drug_ner.prepare_data()

    trained_model = drug_ner.train_model()

    # dataset, predictions, results = drug_ner.evaluate_on_testset(model=trained_model)

    # print(results)
