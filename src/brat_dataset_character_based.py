"""
All credits go to Arne Binder (https://github.com/ArneBinder, arne.binder@dfki.de).
This brat dataloader will soon be integrated into huggingface datasets.

Small modifications by Lisa Raithel to adapt for n2c2 brat format.

WARNING: This is not perfect, since some entities from the annotation file
don't get tokenized properly.

Check out the official DFKI brat dataloader on huggingface (dfki-nlp/brat)!
"""

import glob
import logging
from dataclasses import dataclass
from collections import defaultdict, namedtuple
from os import listdir, path
from typing import Dict, List, Optional
import spacy
import re
import os

import datasets


from standoff2conllmaster.standoff2conll import get_converted_files

from datasets import (
    BuilderConfig,
    DatasetInfo,
    Features,
    Sequence,
    SplitGenerator,
    Value,
)

logger = logging.getLogger(__name__)

TAG_LIST = [
    "NoDisposition",
    "Disposition",
    "Undetermined",
    "Substance",
    "Medication",
    "MEDICATION",
    "substance",
    "CHEM",
    "Drug",
]


@dataclass
class BratConfig(BuilderConfig):
    """BuilderConfig for BRAT."""

    url: str = None  # type: ignore
    description: Optional[str] = None
    citation: Optional[str] = None
    homepage: Optional[str] = None

    subdirectory_mapping: Optional[Dict[str, str]] = None
    file_name_blacklist: Optional[List[str]] = None
    unify_tags: bool = False
    remove_all_except_drug: bool = True
    ann_file_extension: str = "ann"
    txt_file_extension: str = "txt"


class Brat(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIG_CLASS = BratConfig

    def _info(self):
        return DatasetInfo(
            description=self.config.description,
            citation=self.config.citation,
            homepage=self.config.homepage,
            features=Features(
                {
                    # "context": Value("string"),
                    "text": Value("string"),
                    # "tokens": Sequence(Value("string")),
                    "language": Value("string"),
                    "labels": Sequence(Value("string")),
                    # if we unify all relevant tags to "Drug", we only
                    # need those below
                    # "ner_tags": datasets.Sequence(
                    #     datasets.features.ClassLabel(
                    #         names=[
                    #             "O",
                    #             "B-Drug",
                    #             "I-Drug",
                    #         ]
                    #     )
                    # ),
                    # # this is the list of all available token labels
                    # "token_labels": datasets.Sequence(
                    #     datasets.features.ClassLabel(
                    #         names_file="src/utils/entities.txt"
                    #     )
                    # ),
                    # "tokens_per_sentence": Sequence(Sequence(Value("string"))),
                    # "tags_per_sentence": Sequence(Sequence(Value("string"))),
                    # "token_labels_per_sentence": Sequence(Sequence(Value("string"))),
                    "file_name": Value("string"),
                    "spans": Sequence(
                        {
                            "id": Value("string"),
                            "type": Value("string"),
                            "start": Value("int32"),
                            "end": Value("int32"),
                            "text": Value("string"),
                        }
                    ),
                    # "relations": Sequence(
                    #     {
                    #         "id": Value("string"),
                    #         "type": Value("string"),
                    #         "arguments": Sequence(
                    #             {"type": Value("string"), "target": Value("string")}
                    #         ),
                    #     }
                    # ),
                    # "equivalence_relations": Sequence(
                    #     {
                    #         "type": Value("string"),
                    #         "targets": Sequence(Value("string")),
                    #     }
                    # ),
                    # "events": Sequence(
                    #     {
                    #         "id": Value("string"),
                    #         "type": Value("string"),
                    #         "trigger": Value("string"),
                    #         "arguments": Sequence(
                    #             {"type": Value("string"), "target": Value("string")}
                    #         ),
                    #     }
                    # ),
                    # "attributions": Sequence(
                    #     {
                    #         "id": Value("string"),
                    #         "type": Value("string"),
                    #         "target": Value("string"),
                    #         "value": Value("string"),
                    #     }
                    # ),
                    # "normalizations": Sequence(
                    #     {
                    #         "id": Value("string"),
                    #         "type": Value("string"),
                    #         "target": Value("string"),
                    #         "resource_id": Value("string"),
                    #         "entity_id": Value("string"),
                    #     }
                    # ),
                    # "notes": Sequence(
                    #     {
                    #         "id": Value("string"),
                    #         "type": Value("string"),
                    #         "target": Value("string"),
                    #         "note": Value("string"),
                    #     }
                    # ),
                }
            ),
        )

    @staticmethod
    def _get_location(location_string):
        parts = location_string.split(" ")
        assert (
            len(parts) == 2
        ), f"Wrong number of entries in location string. Expected 2, but found: {parts}"
        return {"start": int(parts[0]), "end": int(parts[1])}

    @staticmethod
    def _get_span_annotation(annotation_line):
        """
        example input:
        T1  Organization 0 4    Sony
        """

        _id, remaining, text = annotation_line.split("\t", maxsplit=2)
        _type, locations = remaining.split(" ", maxsplit=1)
        return {
            "id": _id,
            "text": text,
            "type": _type,
            "locations": [Brat._get_location(loc) for loc in locations.split(";")],
        }

    @staticmethod
    def _get_event_annotation(annotation_line):
        """
        example input:
        E1  MERGE-ORG:T2 Org1:T1 Org2:T3
        """
        _id, remaining = annotation_line.strip().split("\t")
        args = [
            dict(zip(["type", "target"], a.split(":"))) for a in remaining.split(" ")
        ]
        return {
            "id": _id,
            "type": args[0]["type"],
            "trigger": args[0]["target"],
            "arguments": args[1:],
        }

    @staticmethod
    def _get_relation_annotation(annotation_line):
        """
        example input:
        R1  Origin Arg1:T3 Arg2:T4
        """

        _id, remaining = annotation_line.strip().split("\t")
        _type, remaining = remaining.split(" ", maxsplit=1)
        args = [
            dict(zip(["type", "target"], a.split(":"))) for a in remaining.split(" ")
        ]
        return {"id": _id, "type": _type, "arguments": args}

    @staticmethod
    def _get_equivalence_relation_annotation(annotation_line):
        """
        example input:
        *   Equiv T1 T2 T3
        """
        _, remaining = annotation_line.strip().split("\t")
        parts = remaining.split(" ")
        return {"type": parts[0], "targets": parts[1:]}

    @staticmethod
    def _get_attribute_annotation(annotation_line):
        """
        example input (binary: implicit value is True, if present, False otherwise):
        A1  Negation E1
        example input (multi-value: explicit value)
        A2  Confidence E2 L1
        """

        _id, remaining = annotation_line.strip().split("\t")
        parts = remaining.split(" ")
        # if no value is present, it is implicitly "true"
        if len(parts) == 2:
            parts.append("true")
        return {
            "id": _id,
            "type": parts[0],
            "target": parts[1],
            "value": parts[2],
        }

    @staticmethod
    def _get_normalization_annotation(annotation_line):
        """
        example input:
        N1  Reference T1 Wikipedia:534366   Barack Obama
        """
        _id, remaining, text = annotation_line.split("\t", maxsplit=2)
        _type, target, ref = remaining.split(" ")
        res_id, ent_id = ref.split(":")
        return {
            "id": _id,
            "type": _type,
            "target": target,
            "resource_id": res_id,
            "entity_id": ent_id,
        }

    @staticmethod
    def _get_note_annotation(annotation_line):
        """
        example input:
        #1  AnnotatorNotes T1   this annotation is suspect
        """
        _id, remaining, note = annotation_line.split("\t", maxsplit=2)
        _type, target = remaining.split(" ")
        return {
            "id": _id,
            "type": _type,
            "target": target,
            "note": note,
        }

    @staticmethod
    def _read_annotation_file(filename):
        """
        reads a BRAT v1.3 annotations file (see https://brat.nlplab.org/standoff.html)
        """

        res = {
            "spans": [],
            "events": [],
            "relations": [],
            "equivalence_relations": [],
            "attributions": [],
            "normalizations": [],
            "notes": [],
        }

        with open(filename, "r", encoding="utf-8") as file:
            for i, line in enumerate(file):
                if len(line.strip()) == 0:
                    continue
                ann_type = line[0]

                # strip away the new line character
                if line.endswith("\n"):
                    line = line[:-1]

                if ann_type == "T":
                    res["spans"].append(Brat._get_span_annotation(line))
                elif ann_type == "E":
                    res["events"].append(Brat._get_event_annotation(line))
                elif ann_type == "R":
                    res["relations"].append(Brat._get_relation_annotation(line))
                elif ann_type == "*":
                    res["equivalence_relations"].append(
                        Brat._get_equivalence_relation_annotation(line)
                    )
                elif ann_type in ["A", "M"]:
                    res["attributions"].append(Brat._get_attribute_annotation(line))
                elif ann_type == "N":
                    res["normalizations"].append(
                        Brat._get_normalization_annotation(line)
                    )
                elif ann_type == "#":
                    res["notes"].append(Brat._get_note_annotation(line))
                else:
                    raise ValueError(
                        f'unknown BRAT annotation id type: "{line}" (from file {filename} @line {i}). '
                        f"Annotation ids have to start with T (spans), E (events), R (relations), "
                        f"A (attributions), or N (normalizations). See "
                        f"https://brat.nlplab.org/standoff.html for the BRAT annotation file "
                        f"specification."
                    )
        return res

    def _unify_tags(tag, remove_all_except_drug=True):
        """Unify the three tag names to one label called `Drug`."""
        for variant in TAG_LIST:
            if tag.endswith(variant):
                return tag.replace(variant, "Drug")

        if remove_all_except_drug:
            return "O"
        return tag

    def _generate_examples(self, files=None, directory=None):
        """Read context (.txt) and annotation (.ann) files."""
        if files is None:
            assert (
                directory is not None
            ), "If files is None, directory has to be provided, but it is also None."
            files = glob.glob(f"{directory}/*.{self.config.ann_file_extension}")
            files_without_ext = sorted([path.splitext(fn)[0] for fn in files])
            # files = sorted([path.splitext(fn) for fn in _files])

        for file_name in files_without_ext:
            mismatch_counter = 0
            annotations = {}
            ann_fn = f"{file_name}.{self.config.ann_file_extension}"
            brat_annotations = Brat._read_annotation_file(ann_fn)
            # get the corresponding txt file
            txt_fn = f"{file_name}.{self.config.txt_file_extension}"

            with open(txt_fn, "r", encoding="utf-8") as read_handle:
                txt_content = read_handle.read()

                annotations["text"] = txt_content

            selection = []
            for span in brat_annotations["spans"]:
                if span["type"] in TAG_LIST:
                    if len(span["locations"]) > 1:
                        sorted_locations = sorted(
                            span["locations"], key=lambda d: d["start"]
                        )
                        span = {
                            "id": span["id"],
                            "text": span["text"],
                            "type": span["type"],
                            "locations": [
                                {
                                    "start": sorted_locations[0]["start"],
                                    "end": sorted_locations[-1]["end"],
                                }
                            ],
                        }

                    if (
                        span["text"]
                        == txt_content[
                            span["locations"][0]["start"] : span["locations"][0]["end"]
                        ]
                    ):
                        selection.append(
                            {
                                "id": span["id"],
                                "text": span["text"],
                                "type": "Drug",
                                "start": span["locations"][0]["start"],
                                "end": span["locations"][0]["end"],
                            }
                        )
                    else:
                        mismatch_counter += 1

            if mismatch_counter > 0:
                print(
                    f"WARNING: Found {mismatch_counter} span mismatches in file {txt_fn}\n"
                )

            labels = ["O"] * len(txt_content)
            spans_sorted = sorted(selection, key=lambda d: d["start"])

            annotations["spans"] = spans_sorted

            # print(txt_content)

            # print(spans_sorted)

            # print(f"len labels: {len(labels)}\nlen text: {len(txt_content)}\n")
            for span in spans_sorted:

                start = span["start"]
                end = span["end"]
                # print(f"start: {start}, end: {end}, len labels: {len(labels)}")
                labels[start] = "B-Drug"
                start += 1
                while start < end:
                    try:
                        # print(f"start: {start}")
                        labels[start] = "I-Drug"
                        start += 1
                    except IndexError:
                        break
            # print(f"final labels: {labels}")

            # print(f"#####################\nANNOTATIONS:\n")

            # for label, char in zip(labels, txt_content):
            #     print(f"'{char}', {label})

            # print("\n############################\n")
            annotations["labels"] = labels

            # the language is encoded at the very beginning of the files,
            # e.g. de_some_corpus.txt
            annotations["language"] = os.path.basename(file_name).split("_")[0]

            annotations["file_name"] = os.path.basename(file_name)

            yield file_name, annotations

            # assert False

        # for i, content_str in enumerate(data):

        #     ann_fn = f"{files_without_ext[i]}.{self.config.ann_file_extension}"
        #     annotations = Brat._read_annotation_file(ann_fn)

        #     # if we unify tags to "Drug", we only need `ner_tags`, otherwise
        #     # we use `token_labels`
        #     # annotations.update(
        #     #     {"tokens": [], "token_labels": [], "ner_tags": [], "language": ""}
        #     # )
        #     # get the corresponding txt file
        #     txt_fn = f"{files_without_ext[i]}.{self.config.txt_file_extension}"

        #     with open(txt_fn, "r", encoding="utf-8") as read_handle:
        #         txt_content = read_handle.read()

        #     # the language is encoded at the very beginning of the files,
        #     # e.g. de_some_corpus.txt
        #     annotations["language"] = os.path.basename(files_without_ext[i]).split("_")[
        #         0
        #     ]

        #     annotations["context"] = txt_content

        #     # split the conll string to lines, i.e. one line looks like this:
        #     # "token \t tag"
        #     lines = content_str.split("\n")
        #     # remove the file name
        #     file_name = lines.pop(0).strip()
        #     annotations["file_name"] = file_name.replace("# doc_id = ", "")
        #     last_line = None

        #     annotations["tokens_per_sentence"] = []
        #     annotations["tags_per_sentence"] = []
        #     annotations["token_labels_per_sentence"] = []

        #     tokens_per_sent = []
        #     tags_per_sent = []
        #     token_labels_per_sent = []

        #     for j, line in enumerate(lines):

        #         if line:
        #             token, tag = line.split("\t")
        #             annotations["tokens"].append(token)
        #             tokens_per_sent.append(token)

        #             # add the non-unified tag
        #             annotations["token_labels"].append(tag)
        #             token_labels_per_sent.append(tag)

        #             unified_tag = Brat._unify_tags(tag)
        #             annotations["ner_tags"].append(unified_tag)
        #             tags_per_sent.append(unified_tag)

        #         # an empty line indicates the end of one sentence
        #         elif line == "" and tags_per_sent:

        #             annotations["tokens_per_sentence"].append(tokens_per_sent)
        #             annotations["tags_per_sentence"].append(tags_per_sent)
        #             annotations["token_labels_per_sentence"].append(
        #                 token_labels_per_sent
        #             )

        #             tokens_per_sent = []
        #             tags_per_sent = []
        #             token_labels_per_sent = []

        #     assert (
        #         len(annotations["ner_tags"])
        #         == len(annotations["tokens"])
        #         == len(annotations["token_labels"])
        #     ), (
        #         f"#tags ({len(annotations['ner_tags'])}) != #tokens "
        #         f"({len(annotations['tokens'])}) != #token labels "
        #         f"({len(annotations['token_labels'])})"
        #     )

        #     assert (
        #         len(annotations["tokens_per_sentence"])
        #         == len(annotations["tags_per_sentence"])
        #         == len(annotations["token_labels_per_sentence"])
        #     ), "#tokenized sentences != #tagged sentences != #token labels per sentence"

        # yield files[i], annotations

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        subdirectory_mapping = self.config.subdirectory_mapping

        # since subclasses of BuilderConfig are not allowed to define
        # attributes without defaults, check here
        assert self.config.url is not None, "data url not specified"

        # if url points to a local directory, just point to that
        if path.exists(self.config.url) and path.isdir(self.config.url):
            data_dir = self.config.url
        # otherwise, download and extract
        else:
            data_dir = dl_manager.download_and_extract(self.config.url)

        logging.info(f"loading from data dir: {data_dir}")
        # if no subdirectory mapping is provided, ...
        if subdirectory_mapping is None:
            # ... use available subdirectories as split names ...
            subdirs = [
                f for f in listdir(data_dir) if path.isdir(path.join(data_dir, f))
            ]
            if len(subdirs) > 0:
                subdirectory_mapping = {subdir: subdir for subdir in subdirs}
            else:
                # ... otherwise, default to a single train split with the base directory
                subdirectory_mapping = {"": "train"}
        return [
            SplitGenerator(
                name=split,
                # These kwargs will be passed to _generate_examples
                gen_kwargs={
                    "directory": path.join(data_dir, subdir),
                },
            )
            for subdir, split in subdirectory_mapping.items()
        ]
