"""..."""
from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_metric
from datetime import datetime
from datetime import datetime
from seqeval.metrics import classification_report
from seqeval.scheme import BILOU
from seqeval.scheme import IOB2
from torch.optim import AdamW
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers import CanineForTokenClassification
from transformers import CanineTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import DefaultDataCollator
from transformers import EarlyStoppingCallback
from transformers import Trainer
from transformers import TrainingArguments
from transformers import get_scheduler
from transformers import set_seed


import argparse
import evaluate
import json
import numpy as np
import os
import torch
import wandb


from bio2brat import convert
from utils import utils
from utils.early_stopping import EarlyStopping

os.environ["HF_DATASETS_OFFLINE"] = "1"

current_time = datetime.now().strftime("%y_%m_%d_%H_%M")
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


with open("src/utils/lang_dict.json", "r") as read_handle:
    lang2id = json.load(read_handle)
    id2lang = {value: key for key, value in lang2id.items()}


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
        self.out_dir = self.config["out_dir"]

        # will be filled when preparing data
        self.label_list = []
        self.label2id = {}
        self.id2label = {}

        self.metric = evaluate.load("seqeval")
        self.fair_eval = evaluate.load("hpi-dhc/FairEval", suffix=False, scheme="IOB2")

    def get_tokenizer(self, model=False):
        """Load a pre-trained tokenizer."""
        if not model:
            model = self.config["model_name"]

        if self.config["model_name"].startswith("google/canine"):
            self.tokenizer = CanineTokenizer.from_pretrained("google/canine-c")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model, use_fast=True, add_prefix_space=True, strip_accent=False
            )

    def get_model(self, model=None):
        """Load a pre-trained model."""
        if model is None:
            model = self.config["model_name"]

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

            return tokenized_inputs
        else:

            # sub-tokenize input sentences and convert to IDs
            tokenized_inputs = self.tokenizer(
                chunked_tokens,
                is_split_into_words=True,
                truncation=True,
                # max_length=510,
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

    def sort_by_language(self, predictions, labels, languages):
        """Make one set of predictions-labels for each language."""
        by_language = {
            lang: {"predictions": [], "true_labels": []} for lang in languages
        }

        for pred, label, language in zip(predictions, labels, languages):
            by_language[language]["predictions"].append(pred)
            by_language[language]["true_labels"].append(label)

        return by_language

    def compute_metrics(self, predictions, labels, languages):
        """..."""
        print("Computing metrics ... ")
        # predictions, labels = p
        # predictions = np.argmax(predictions, axis=2)

        # Remove ignored index (special tokens)
        cleaned_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        sorted_by_language = self.sort_by_language(
            predictions=cleaned_predictions, labels=true_labels, languages=languages
        )
        for language, outputs in sorted_by_language.items():

            results_trad_per_lang = self.metric.compute(
                predictions=outputs["predictions"],
                references=outputs["true_labels"],
                suffix=False,
            )

            wandb.log({id2lang[language]: results_trad_per_lang})

            try:
                results_fair_per_lang = self.fair_eval.compute(
                    predictions=outputs["predictions"],
                    references=outputs["true_labels"],
                    mode="fair",
                    error_format="count",
                )
                cls_report_per_lang = classification_report(
                    y_true=outputs["true_labels"],
                    y_pred=outputs["predictions"],
                    # output_dict=True,
                )

                wandb.log({id2lang[language]: results_fair_per_lang})

            except ValueError:
                print(
                    f"Warning: could not get results for language '{id2lang[language]}'"
                )
                pass

        # compute results over all entities, independent of language
        # results_trad = self.metric.compute(
        #     predictions=cleaned_predictions, references=true_labels, suffix=False
        # )

        cls_report_dict = classification_report(
            y_true=true_labels, y_pred=cleaned_predictions, output_dict=True
        )
        print(cls_report_dict)
        wandb.log({"traditional": cls_report_dict})

        try:
            results_fair = self.fair_eval.compute(
                predictions=cleaned_predictions,
                references=true_labels,
                mode="fair",
                error_format="count",
            )
            wandb.log({"fair": results_fair})

        except ValueError:
            pass

        return cls_report_dict

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

    # def train_and_evaluate_model(self, model=None):
    #     """..."""
    #     self.get_model(model=model)

    #     model = self.config["model_name"]
    #     model_name = model.split("/")[-1]

    #     if self.debug:
    #         epochs = 10
    #         eval_steps = 20  # 500
    #         save_steps = 20

    #     else:
    #         epochs = self.config["epochs"]
    #         eval_steps = 500  # 500
    #         save_steps = 500

    #     out_dir = os.path.join(
    #         self.out_dir,
    #         f"{model_name}-finetuned-ner_unifytags_{self.config['unify_tags']}_{self.time}",
    #     )

    #     args = TrainingArguments(
    #         output_dir=out_dir,
    #         evaluation_strategy="steps",
    #         save_strategy="steps",
    #         eval_steps=eval_steps,  # 500
    #         save_steps=save_steps,
    #         learning_rate=self.config["learning_rate"],
    #         per_device_train_batch_size=self.config["batch_size"],
    #         per_device_eval_batch_size=self.config["batch_size"],
    #         num_train_epochs=epochs,
    #         weight_decay=0.01,
    #         push_to_hub=False,
    #         seed=self.config["seed"],
    #         load_best_model_at_end=True,
    #         metric_for_best_model="overall_f1",
    #         greater_is_better=True,
    #         save_total_limit=1,
    #         report_to="wandb",
    #     )

    #     # batches processed examples together while applying padding to make
    #     # them all the same size
    #     # if self.config["model_name"] != "google/canine-c":
    #     data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

    #     trainer = Trainer(
    #         self.model,
    #         args=args,
    #         train_dataset=self.tokenized_datasets["train"],
    #         eval_dataset=self.tokenized_datasets["dev"],
    #         data_collator=data_collator,
    #         tokenizer=self.tokenizer,
    #         compute_metrics=self.compute_metrics,
    #         callbacks=[
    #             EarlyStoppingCallback(early_stopping_patience=self.config["patience"])
    #         ],
    #     )
    #     # else:
    #     # trainer = Trainer(
    #     #     self.model,
    #     #     args=args,
    #     #     train_dataset=self.datasets["train"],
    #     #     eval_dataset=self.datasets["dev"],
    #     #     # data_collator=data_collator,
    #     #     # tokenizer=self.tokenizer,
    #     #     compute_metrics=self.compute_metrics,
    #     #     callbacks=[
    #     #         EarlyStoppingCallback(
    #     #             early_stopping_patience=self.config["patience"]
    #     #         )
    #     #     ],
    #     # )
    #     trainer.train()
    #     trainer.evaluate()

    #     return trainer

    def get_scheduler_and_optimizer(self, wu_steps, train_steps):
        """..."""
        if self.config["optimizer"] == "adamw":
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        elif self.config["optimizer"] == "radam":
            optimizer = RAdam(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )
        else:
            optimizer = AdamW(
                self.model.parameters(),
                lr=self.config["learning_rate"],
                weight_decay=self.config["weight_decay"],
            )

        lr_scheduler = get_scheduler(
            name="linear",
            optimizer=optimizer,
            num_warmup_steps=wu_steps,
            num_training_steps=train_steps,
        )

        return lr_scheduler, optimizer

    def train_and_evaluate_model(self, model=None):
        """..."""
        self.get_model(model=model)

        model_name = self.config["model_name"]
        model_name = model_name.split("/")[-1]

        self.tokenized_datasets.set_format("torch")

        data_collator = DataCollatorForTokenClassification(self.tokenizer, padding=True)

        train_dataloader = DataLoader(
            self.tokenized_datasets["train"],
            shuffle=True,
            batch_size=self.config["batch_size"],
            collate_fn=data_collator,
        )
        eval_dataloader = DataLoader(
            self.tokenized_datasets["dev"],
            batch_size=self.config["batch_size"],
            collate_fn=data_collator,
        )

        es = EarlyStopping(patience=self.config["patience"], mode="max")

        if self.debug:
            num_epochs = 10
            eval_steps = 10  # 500
            save_steps = 10
            warmup_steps = 10

        else:
            num_epochs = self.config["epochs"]
            eval_steps = 500  # 500
            save_steps = 500
            warmup_steps = self.config.get("warmup_steps", 200)

        num_training_steps = num_epochs * len(train_dataloader)

        lr_scheduler, optimizer = self.get_scheduler_and_optimizer(
            wu_steps=warmup_steps, train_steps=num_training_steps
        )

        self.model.to(device)

        progress_bar = tqdm(range(num_training_steps))
        best_macro_f1 = 0.0
        best_model = None

        self.model.train()

        for epoch in range(num_epochs):

            for batch in train_dataloader:

                batch = {k: v.to(device) for k, v in batch.items()}

                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                }

                outputs = self.model(**inputs)
                loss = outputs.loss
                wandb.log({"train/loss": loss})
                loss.backward()

                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                progress_bar.update(1)

            self.model.eval()

            eval_predictions = []
            eval_languages = []
            eval_true_labels = []

            for batch in eval_dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}

                inputs = {
                    "input_ids": batch["input_ids"],
                    "attention_mask": batch["attention_mask"],
                    "labels": batch["labels"],
                }

                with torch.no_grad():
                    outputs = self.model(**inputs)

                eval_loss = outputs.loss
                wandb.log({"eval/loss": eval_loss})

                logits = outputs.logits
                predictions = (
                    torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
                )
                true_labels = batch["labels"].detach().cpu().numpy().tolist()
                languages = batch["language"].detach().cpu().numpy().tolist()

                eval_predictions.extend(predictions)
                eval_true_labels.extend(true_labels)
                eval_languages.extend(languages)

                # metric.add_batch(predictions=predictions, references=batch["labels"])

            results = self.compute_metrics(
                predictions=eval_predictions,
                labels=eval_true_labels,
                languages=eval_languages,
            )

            current_macro_f1 = results["macro avg"]["f1-score"]
            wandb.log({"macro_f1": current_macro_f1})

            # save best F1 and model
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
                best_model = self.model
                wandb.log(
                    {
                        "best_macro_f1": best_macro_f1,
                        "epoch_of_best_f1": epoch,
                    }
                )

                model_id = os.path.join(
                    self.out_dir,
                    f"weights_{model_name.replace('/', '-')}_{current_time}.pth",
                )
                wandb.log({"model_id": model_id})
                
                if self.config["save_best_model"]:
                    torch.save(best_model.state_dict(), model_id)

            if es.step(current_macro_f1):
                print(f"Stopping training with F1 of {current_macro_f1}.")
                break

        return True

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

        if self.config["unify_tags"]:
            self.label_list = datasets["train"].features["ner_tags"].feature.names
        else:
            self.label_list = datasets["train"].features["token_labels"].feature.names

        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}

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

        # print(self.datasets)

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

        # remove features that are not necessary for training
        self.tokenized_datasets = self.tokenized_datasets.remove_columns(
            [
                col
                for col in self.tokenized_datasets["train"].features
                if col
                not in [
                    "labels",
                    "language",
                    "input_ids",
                    "token_type_ids",
                    "attention_mask",
                ]
            ]
        )

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
