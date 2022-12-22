#!/usr/bin/env python
"""

# Script to convert a column-based BIO-formatted entity-tagged file
# into standoff with reference to the original text.


08/03/2022:
Lisa Raithel: script taken from the original brat implementation (BIOtoStandoff.py)
and modified for transformer output sequences.


"""

import os
import re
import sys
import unicodedata


class taggedEntity:
    def __init__(self, startOff, endOff, eType, idNum, fullText):
        self.startOff = startOff
        self.endOff = endOff
        self.eType = eType
        self.idNum = idNum
        self.fullText = fullText

        self.eText = fullText[startOff:endOff]

    def __str__(self):
        return "T%d\t%s %d %d\t%s" % (
            self.idNum,
            self.eType,
            self.startOff,
            self.endOff,
            self.eText,
        )

    def check(self):
        # sanity checks: the string should not contain newlines and
        # should be minimal wrt surrounding whitespace
        if "\n" in self.eText:
            print("ERROR: newline in entity: '%s'" % self.eText)
        # assert "\n" not in self.eText, "ERROR: newline in entity: '%s'" % self.eText

        # if self.eText != self.eText.strip():
        #     print(f"ERROR: entity contains extra whitespace: '{self.eText}'")
        assert self.eText == self.eText.strip(), (
            "ERROR: entity contains extra whitespace: '%s'" % self.eText
        )


def BIO_to_standoff(BIOtext, reftext, tokenidx=2, tagidx=-1):
    BIOlines = BIOtext.split("\n")
    return BIO_lines_to_standoff(BIOlines, reftext, tokenidx, tagidx)


next_free_id_idx = 1


def BIO_lines_to_standoff(BIOlines, reftext, tokenidx=2, tagidx=-1):
    global next_free_id_idx

    taggedTokens = []

    BIOlines = [unicodedata.normalize("NFKD", token) for token in BIOlines]
    reftext = unicodedata.normalize("NFKD", reftext)

    ri, bi = 0, 0
    while ri < len(reftext):
        if bi >= len(BIOlines):
            print("Warning: received BIO didn't cover given text", file=sys.stderr)
            break

        BIOline = BIOlines[bi]

        if re.match(r"^\s*$", BIOline):
            # the BIO has an empty line (sentence split); skip
            bi += 1
        else:
            # assume tagged token in BIO. Parse and verify
            fields = BIOline.split("\t")
            try:
                tokentext = fields[tokenidx]
            except BaseException:
                print(
                    "Error: failed to get token text "
                    "(field %d) on line: %s" % (tokenidx, BIOline),
                    file=sys.stderr,
                )
                raise

            try:
                tag = fields[tagidx]
            except BaseException:
                print(
                    "Error: failed to get token text "
                    "(field %d) on line: %s" % (tagidx, BIOline),
                    file=sys.stderr,
                )
                raise

            m = re.match(r"^([BIO])((?:-[A-Za-z0-9_-]+)?)$", tag)
            assert m, "ERROR: failed to parse tag '%s'" % tag
            ttag, ttype = m.groups()

            # strip off starting "-" from tagged type
            if len(ttype) > 0 and ttype[0] == "-":
                ttype = ttype[1:]

            # sanity check
            assert (ttype == "" and ttag == "O") or (
                ttype != "" and ttag in ("B", "I")
            ), ("Error: tag/type mismatch %s" % tag)

            # go to the next token on reference; skip whitespace
            while ri < len(reftext) and reftext[ri].isspace():
                ri += 1

            # verify that the text matches the original
            # assert (
            #     reftext.lower()[ri : ri + len(tokentext)] == tokentext.lower()
            # ), f"ERROR: text mismatch: reference '{reftext.lower()[ri : ri + len(tokentext)].encode('UTF-8')}' tagged '{tokentext.lower().encode('UTF-8')}\n{reftext[ri - 100 : ri + len(tokentext) + 100]}'"#;\n{reftext[ri - 100 : ri + len(tokentext) + 100]}\n{BIOlines[bi - 1]}\n{BIOline}\n{BIOlines[bi+1]}" #\n{reftext}\n{BIOlines}
            if reftext.lower()[ri : ri + len(tokentext)] != tokentext.lower():
                print(
                    f"ERROR: text mismatch: reference '{reftext.lower()[ri : ri + len(tokentext)].encode('UTF-8')}' tagged '{tokentext.lower().encode('UTF-8')}'"
                )  # ;\n{reftext[ri - 100 : ri + len(tokentext) + 100]}\n{BIOlines[bi - 1]}\n{BIOline}\n{BIOlines[bi+1]}" #\n{reftext}\n{BIOlines}
                break

            # store tagged token as (begin, end, tag, tagtype) tuple.
            taggedTokens.append((ri, ri + len(tokentext), ttag, ttype))

            # skip the processed token
            ri += len(tokentext)
            bi += 1

            # ... and skip whitespace on reference
            while ri < len(reftext) and reftext[ri].isspace():
                ri += 1

    # if the remaining part either the reference or the tagged
    # contains nonspace characters, something's wrong
    if (
        len([c for c in reftext[ri:] if not c.isspace()]) != 0
        or len([c for c in BIOlines[bi:] if not re.match(r"^\s*$", c)]) != 0
    ):
        # assert (
        #     False
        # ), "ERROR: failed alignment: '%s' remains in reference, " "'%s' in tagged" % (
        #     reftext[ri:],
        #     BIOlines[bi:],
        # )
        print(
            f"ERROR: failed alignment: '{reftext[ri:]}' remains in reference, '{BIOlines[bi:]}' in tagged"
        )

    standoff_entities = []

    # cleanup for tagger errors where an entity begins with a
    # "I" tag instead of a "B" tag
    revisedTagged = []
    prevTag = None
    for startoff, endoff, ttag, ttype in taggedTokens:
        if prevTag == "O" and ttag == "I":
            print('Note: rewriting "I" -> "B" after "O"', file=sys.stderr)
            ttag = "B"
        revisedTagged.append((startoff, endoff, ttag, ttype))
        prevTag = ttag
    taggedTokens = revisedTagged

    # cleanup for tagger errors where an entity switches type
    # without a "B" tag at the boundary
    revisedTagged = []
    prevTag, prevType = None, None
    for startoff, endoff, ttag, ttype in taggedTokens:
        if prevTag in ("B", "I") and ttag == "I" and prevType != ttype:
            print('Note: rewriting "I" -> "B" at type switch', file=sys.stderr)
            ttag = "B"
        revisedTagged.append((startoff, endoff, ttag, ttype))
        prevTag, prevType = ttag, ttype
    taggedTokens = revisedTagged

    prevTag, prevEnd = "O", 0
    currType, currStart = None, None
    for startoff, endoff, ttag, ttype in taggedTokens:
        if prevTag != "O" and ttag != "I":
            # previous entity does not continue into this tag; output
            assert currType is not None and currStart is not None, "ERROR in %s" % fn

            # In case there is a new line in the entity, that probably means
            # that these are two separate entities. Split them, and add them
            # separately
            if "\n" in reftext[currStart:prevEnd]:
                ent = reftext[currStart:prevEnd]
                ent_list = re.split("(\W)", ent)

                for i, e in enumerate(ent_list):
                    if e not in ("\n", "", "\t", " "):
                        tg = taggedEntity(
                            currStart,
                            currStart + len(e),
                            currType,
                            next_free_id_idx,
                            reftext,
                        )
                        standoff_entities.append(tg)

                        next_free_id_idx += 1
                        currStart = currStart + len(e)
                    else:
                        currStart = currStart + len(e)

            else:
                standoff_entities.append(
                    taggedEntity(
                        currStart, prevEnd, currType, next_free_id_idx, reftext
                    )
                )

                next_free_id_idx += 1

            # reset current entity
            currType, currStart = None, None

        elif prevTag != "O":
            # previous entity continues ; just check sanity
            assert ttag == "I", "ERROR in %s" % fn
            assert (
                currType == ttype
            ), "ERROR: entity of type '%s' continues " "as type '%s'" % (
                currType,
                ttype,
            )

        if ttag == "B":
            # new entity starts
            currType, currStart = ttype, startoff

        prevTag, prevEnd = ttag, endoff

    # if there's an open entity after all tokens have been processed,
    # we need to output it separately
    if prevTag != "O":
        standoff_entities.append(
            taggedEntity(currStart, prevEnd, currType, next_free_id_idx, reftext)
        )
        next_free_id_idx += 1

    for e in standoff_entities:
        e.check()

    return standoff_entities


RANGE_RE = re.compile(r"^(-?\d+)-(-?\d+)$")


def parse_indices(idxstr):
    # parse strings of forms like "4,5" and "6,8-11", return list of
    # indices.
    indices = []
    for i in idxstr.split(","):
        if not RANGE_RE.match(i):
            indices.append(int(i))
        else:
            start, end = RANGE_RE.match(i).groups()
            for j in range(int(start), int(end)):
                indices.append(j)
    return indices


def convert(text_file, model_predictions, tokens):
    """..."""
    with open(text_file, "r") as textf:
        text = textf.read()

    bio_str = ""

    for token_list, pred_list in zip(tokens, model_predictions):
        for token, pred in zip(token_list, pred_list):
            bio_str += f"{token}\t{pred}\n"
        bio_str += "\n"

    # print(bio_str)
    # print(text)
    try:
        reconstructed_brat = BIO_to_standoff(
            BIOtext=bio_str, reftext=text, tokenidx=0, tagidx=1
        )
    except AssertionError as e:
        print("Could not reconstruct.")
        print(e)
        reconstructed_brat = ""
        raise e

    return reconstructed_brat, bio_str


if __name__ == "__main__":
    from standoff2conllmaster.standoff2conll import get_converted_files
    from collections import namedtuple

    ann_file = "/home/lisa/projects/n2c2_2022/data/brat_format/n2c2Track1TrainingData-v3/data_v3/train/105-02.ann"
    txt_file = "/home/lisa/projects/n2c2_2022/data/brat_format/n2c2Track1TrainingData-v3/data_v3/train/105-02.txt"

    options = namedtuple(
        "OPTIONS",
        [
            "singletype",
            "asciify",
            "char_offsets",
            "no_sentence_split",
            "include_docid",
            "discont_rule",
            "overlap_rule",
            "types",
            "exclude",
            "tokenization",
            "tagset",
        ],
    )
    options.singletype = None
    options.asciify = False
    options.char_offsets = False
    options.no_sentence_split = True
    options.discont_rule = "full-span"
    options.overlap_rule = "keep-longer"
    options.types = None
    options.exclude = None
    options.tokenization = "default"
    options.tagset = None  # default = BIO

    # returns a conll-style string per file
    data = get_converted_files([ann_file], options)

    lines = data[0].split("\n")
    lines.pop(0).strip()

    tokens = []
    tags = []
    for line in lines:

        if line:
            token, tag = line.split("\t")
            tokens.append(token)
            if tag.endswith("NoDisposition"):
                tag = tag.replace("NoDisposition", "Drug")
            elif tag.endswith("Disposition"):
                tag = tag.replace("Disposition", "Drug")
            elif tag.endswith("Undetermined"):
                tag = tag.replace("Undetermined", "Drug")
            tags.append(tag)

    # predictions = ["O"] * len(tokens)

    output_dir = "data/predicted_annotations/"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for prediction, tokens, txt_file in zip([tags], [tokens], [txt_file]):

        assert len(tokens) == len(prediction)
        brat_anno = convert(
            text_file=txt_file, model_predictions=prediction, tokens=tokens
        )
        file_name = os.path.basename(txt_file).split(".txt")[0]
        path = os.path.join(output_dir, f"{file_name}_predicted.ann")

        with open(path, "w") as write_handle:
            for line in brat_anno:
                write_handle.write(str(line))
                write_handle.write("\n")
