"""..."""
# from datasets import DatasetDict
from datasets import load_dataset
from datasets import load_metric
from datetime import datetime
from seqeval.metrics import classification_report
from torch.optim import AdamW
from torch.optim import RAdam
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import CanineForTokenClassification
from transformers import CanineTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import get_scheduler

# from transformers import set_seed


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

    def __init__(self, config, mode="train"):
        super(DrugNER, self).__init__()

        self.mode = mode
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

        self.time = datetime.now().strftime("%d_%m_%y_%H_%M")

        self.data_url = self.config["data_url"]

        model_name = self.config["model_name"].split("/")[-1]
        if self.mode == "train":
            self.out_dir = os.path.join(
                self.config["out_dir"],
                f"checkpoint_{model_name.replace('/', '-')}_{current_time}",
            )
            os.makedirs(self.out_dir, exist_ok=True)

        # will be filled when preparing data
        self.label_list = []
        self.label2id = {}
        self.id2label = {}

        # load the traditional and "fair" eval metrics
        self.trad_eval = evaluate.load("seqeval")
        self.fair_eval = evaluate.load("hpi-dhc/FairEval", suffix=False, scheme="IOB2")

    def get_tokenizer(self, model=False):
        """Load a pre-trained tokenizer."""
        print(f"get tokenizer: model: {model}")
        if not model:
            model = self.config["model_name"]

        self.tokenizer = CanineTokenizer.from_pretrained(
            model,
            use_fast=True,
            strip_accent=False,
            # bos_token="\ue000",
            # eos_token="\ue001",
        )

    def get_model(self, model=None):
        """Load a pre-trained model (fine-tuned or not)."""
        if model is None:
            model = self.config["model_name"]

        self.model = CanineForTokenClassification.from_pretrained(
            self.config["model_name"],
            id2label=self.id2label,
            label2id=self.label2id,
        )

    def tokenize_and_align_labels(self, documents, label_all_tokens=True):
        """Align the labels with the IDs.

        In this case (using a char-based model), we only need to prepend and append
        BOS and EOS markers, and add -100 for padding characters
        """
        # max_length = self.config["window_size"]
        max_length = 2048

        tokenized_inputs = self.tokenizer(
            documents["texts"],
            truncation=True,
            # max_length=max_length,
            padding="longest",
            return_tensors="pt",
            is_split_into_words=False,
        )

        tokenized_docs = tokenized_inputs["input_ids"]
        labels = documents["labels"]
        adapted_labels = []

        for label_list, character_id_list in zip(labels, tokenized_docs):
            # add -100 for the start of the document
            adapted_labels_per_doc = [-100]
            char_counter = 0

            for lab, char_id in zip(label_list, character_id_list[1:]):
                # 57344 is the start id
                # We set the label for all characters of each token
                if char_id == 57344:
                    adapted_labels_per_doc.append(-100)
                elif char_id == 0:
                    adapted_labels_per_doc.append(-100)
                else:
                    adapted_labels_per_doc.append(self.label2id[lab])

                # For the other tokens in a word, we set the label to
                # either the current label or -100, depending on the
                # label_all_tokens flag.
                # else:
                char_counter += 1
            # add an end-of-sentence marker for the labels
            adapted_labels_per_doc.append(-100)
            char_counter += 1

            # add as many -100 label IDs as possible, but cut off the
            # sequence after 2048 elements (max sequence length)
            adapted_labels_per_doc.extend(
                [-100] * (len(character_id_list[char_counter + 1 :] + 1))
            )

            adapted_labels_per_doc = adapted_labels_per_doc[:max_length]

            adapted_labels.append(adapted_labels_per_doc)

            assert len(adapted_labels_per_doc) == len(
                character_id_list
            ), f"lengths do not match: label: {len(adapted_labels_per_doc)} vs. characters: {len(character_id_list)}"

        # for all_labels, all_texts in zip(adapted_labels, tokenized_docs):

        #     for lab, char in zip(all_labels, all_texts):
        #         print(char, lab)

        #     print("\n")

        tokenized_inputs["labels"] = adapted_labels

        return tokenized_inputs

    def compute_metrics(self, predictions, labels, languages):
        """Compute metrics for sequence labeling.

        Collect different metrics:

        traditional: HF seqeval implementation (https://huggingface.co/spaces/evaluate-metric/seqeval)
        fair: HF faireval implementation (https://huggingface.co/spaces/hpi-dhc/FairEval)

        We further collect the *best macro F1* score, calculated using the
        traditional metric.

        We report:

        - traditional
            - all scores ("all_trad")
            - per language: de, en, fr (e.g. "de_trad")
        - fair
            - all scores ("all_fair")
            - per language: de, en, fr (e.g. "de_fair")
        """
        print("Computing metrics ... ")

        # Remove ignored index (special tokens)
        cleaned_predictions = [
            [self.label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # print(
        #     f"cleaned_predictions: {cleaned_predictions}, length: {len(cleaned_predictions)}"
        # )

        true_labels = [
            [self.label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        # print(f"true labels: {true_labels}, length: {len(true_labels)}")

        sorted_by_language = utils.sort_by_language(
            predictions=cleaned_predictions, labels=true_labels, languages=languages
        )
        for language, outputs in sorted_by_language.items():

            results_trad_per_lang = self.trad_eval.compute(
                predictions=outputs["predictions"],
                references=outputs["true_labels"],
                suffix=False,
            )

            print(f"{id2lang[language]}: {results_trad_per_lang}")

            wandb.log({f"{id2lang[language]}_trad": results_trad_per_lang})

            try:
                results_fair_per_lang = self.fair_eval.compute(
                    predictions=outputs["predictions"],
                    references=outputs["true_labels"],
                    mode="fair",
                    error_format="count",
                )

                wandb.log({f"{id2lang[language]}_fair": results_fair_per_lang})

            except ValueError:
                print(
                    f"Warning: could not get results for language '{id2lang[language]}'"
                )
                pass

        cls_report_dict = classification_report(
            y_true=true_labels, y_pred=cleaned_predictions, output_dict=True
        )
        wandb.log({"all_trad": cls_report_dict})

        try:
            results_fair = self.fair_eval.compute(
                predictions=cleaned_predictions,
                references=true_labels,
                mode="fair",
                error_format="count",
            )
            wandb.log({"all_fair": results_fair})

        except ValueError:
            pass

        # return the traditional classification report over all languages
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

        results = self.trad_eval.compute(
            predictions=converted_predictions, references=true_labels, suffix=False
        )
        wandb.log({"final_eval": results})

        return self.tokenized_datasets["test"], converted_predictions, results

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
            print("No optimizer given, using AdamW.\n")
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

        # utils.print_gpu_utilization()

        model_name = self.config["model_name"].split("/")[-1]

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
            eval_steps = 10
            save_steps = 10
            warmup_steps = 10

        else:
            num_epochs = self.config["epochs"]
            eval_steps = 500
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

        for epoch in range(num_epochs):

            wandb.log({"current_epoch": epoch})

            self.model.train()

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

                predicted_char_class_ids = (
                    torch.argmax(logits, dim=-1).detach().cpu().numpy().tolist()
                )
                # print(f"predicted char class ids: {predicted_char_class_ids}")

                # Note that tokens are classified rather then input words which means that
                # there might be more predicted token classes than words.
                # Multiple token classes might account for the same word

                true_labels = batch["labels"].detach().cpu().numpy().tolist()
                languages = batch["languages"].detach().cpu().numpy().tolist()

                # eval_predictions.extend(predictions)
                eval_predictions.extend(predicted_char_class_ids)

                eval_true_labels.extend(true_labels)
                eval_languages.extend(languages)

            results = self.compute_metrics(
                predictions=eval_predictions,
                labels=eval_true_labels,
                languages=eval_languages,
            )

            current_macro_f1 = results["macro avg"]["f1-score"]
            wandb.log({"macro_f1_trad": current_macro_f1})

            # save best F1 and model
            if current_macro_f1 > best_macro_f1:
                best_macro_f1 = current_macro_f1
                best_model = self.model
                wandb.log(
                    {
                        "best_macro_f1_trad": best_macro_f1,
                        "epoch_of_best_f1_trad": epoch,
                    }
                )

                if self.config["save_best_model"]:
                    # model_path = os.path.join(
                    #     self.out_dir,
                    #     f"checkpoint_{model_name.replace('/', '-')}_{current_time}.pth",
                    # )
                    wandb.log({"model_id": self.out_dir})

                    self.tokenizer.save_pretrained(self.out_dir)

                    best_model.save_pretrained(self.out_dir)

            # utils.print_gpu_utilization()

            if es.step(current_macro_f1):
                print(f"Stopping training with F1 of {current_macro_f1}.")
                break

        return best_model

    def prepare_data(self, model=False, download_mode="force_redownload"):
        """..."""
        path_to_loader = "src/brat_dataset_character_based.py"

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
            cache_dir=self.config["cache_dir"],
        )

        print(self.config["data_url"])

        # dataset = load_dataset('dfki-nlp/brat', **kwargs)

        self.label_list = ["O", "B-Drug", "I-Drug"]

        self.label2id = {l: i for i, l in enumerate(self.label_list)}
        self.id2label = {i: l for i, l in enumerate(self.label_list)}

        # print(datasets)

        print(self.label2id)

        datasets = datasets.map(
            utils.map_languages, batched=True, remove_columns=["language"]
        )
        # split the documents in chunks
        datasets = datasets.map(
            utils.chunk_documents_into_windows,
            fn_kwargs={
                "window_size": self.config["window_size"],
            },
            batched=True,
            remove_columns=datasets["train"].column_names,
        )
        # print(datasets["train"]["texts"])
        # print(datasets["train"]["labels"])

        # # combine the lists of lists of tokens to lists of tokens (should be done in mapping above)
        # self.datasets = datasets.map(
        #     utils.combine_x_sentences,
        #     batched=True,
        #     # those two columns are not really removed but replaced by the
        #     # same column names
        #     remove_columns=["chunks_tokens", "chunks_tags"],
        # )

        self.get_tokenizer(model=model)

        self.tokenized_datasets = datasets.map(
            self.tokenize_and_align_labels,
            batched=True,
            fn_kwargs={"label_all_tokens": True},
        )

        if self.mode == "train":
            # remove features that are not necessary for training
            self.tokenized_datasets = self.tokenized_datasets.remove_columns(
                [
                    col
                    for col in self.tokenized_datasets["train"].features
                    if col
                    not in [
                        "labels",
                        "languages",
                        "input_ids",
                        "token_type_ids",
                        "attention_mask",
                        # "texts",
                        # "file_ids",
                    ]
                ]
            )
        else:
            # remove features that are not necessary for training
            self.tokenized_datasets = self.tokenized_datasets.remove_columns(
                [
                    col
                    for col in self.tokenized_datasets["train"].features
                    if col
                    not in [
                        "labels",
                        "languages",
                        "input_ids",
                        "token_type_ids",
                        "attention_mask",
                        "texts",
                        "file_ids",
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

    # utils.print_gpu_utilization()

    drug_ner = DrugNER(args.config)

    drug_ner.prepare_data()

    trained_model = drug_ner.train_model()

    # dataset, predictions, results = drug_ner.evaluate_on_testset(model=trained_model)

    # print(results)
