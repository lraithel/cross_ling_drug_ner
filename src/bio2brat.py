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

    # BIOlines = [unicodedata.normalize("NFKD", token) for token in BIOlines]
    # reftext = unicodedata.normalize("NFKD", reftext)

    # print(BIOlines[:150])
    # print(reftext[:1000])

    # ref and bio indices
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
            if reftext.lower()[ri : ri + len(tokentext)] != tokentext.lower():
                print(
                    f"WARNING: text mismatch: reference '{reftext.lower()[ri : ri + len(tokentext) + 20].encode('UTF-8')}' tagged '{tokentext.lower().encode('UTF-8')}'"
                )
                normalized_ref = unicodedata.normalize(
                    "NFKD", reftext[ri : ri + len(tokentext)]
                )
                normalized_tokentext = unicodedata.normalize("NFKD", tokentext)
                # if the normalized versions do not correspond, raise an error

                if normalized_ref != normalized_tokentext:

                    # try another hack, strings might be including each other
                    if len(normalized_ref) >= len(normalized_tokentext):
                        start_of_tokentext = normalized_ref.find(normalized_tokentext)
                        ri = start_of_tokentext
                    else:
                        start_of_ref = normalized_tokentext.find(normalized_ref)
                        ri = start_of_ref

                    # assert (
                    #     False
                    # ), f"WARNING: text mismatch: reference '{normalized_ref}' tagged '{normalized_tokentext}'"

                    # # continue
                    # # assert False, "text mismatch"
                    # # find position of tokentext in reftext -- if it is not
                    # # too far away, skip to it, i.e. increase ri
                    # current_ref = reftext[ri : ri + len(tokentext) + 20]
                    # start_current_ref = reftext.find(current_ref)
                    # # print(f"start current ref: {start_current_ref}")
                    # # find occurrence of token text in window
                    # window_size = 400
                    # new_ri = -1
                    # while new_ri == -1:
                    #     # print(f"window size: {window_size}")
                    #     if window_size <= 0:
                    #         break
                    #     new_ri = reftext.find(
                    #         tokentext, start_current_ref + window_size
                    #     )
                    #     # print(f"new_ri: {new_ri}")
                    #     # decrease the window size if the
                    #     window_size -= 50

                    # # print(f"UPDATING ri: {new_ri}")
                    # # update the current ri
                    # if new_ri != -1:
                    #     ri = new_ri
                    # print(f"appending {(ri, ri + len(tokentext), ttag, ttype)}")

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
        # assert False, "failed alignment"

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
                ent_list = re.split(r"(\W)", ent)

                for i, e in enumerate(ent_list):
                    if e not in ("\n", "", "\t", " "):
                        tg = taggedEntity(
                            currStart,
                            currStart + len(e),
                            currType,
                            next_free_id_idx,
                            reftext,
                        )

                        if len(reftext[currStart:prevEnd]) > 2:
                            # print("\nWEIRD entitiy")
                            # print(
                            #     currStart,
                            #     prevEnd,
                            #     currType,
                            #     next_free_id_idx,
                            #     reftext[currStart:prevEnd],
                            # )
                            # print("~~~~~~~~~~~\n")

                            standoff_entities.append(tg)

                        next_free_id_idx += 1
                        currStart = currStart + len(e)
                    else:
                        currStart = currStart + len(e)

            # check for trailing white space
            elif reftext[currStart:prevEnd] != reftext[currStart:prevEnd].strip():
                # use split to get the single white spaces
                ent_list = reftext[currStart:prevEnd].split(" ")
                ent = reftext[currStart:prevEnd].strip()

                num_beginning = 0
                num_end = 0
                hit_token = False

                for element in ent_list:
                    if element == " ":
                        if not hit_token:
                            num_beginning += 1
                        if hit_token:
                            break
                    elif element != " ":
                        hit_token = True

                for element in ent_list[::-1]:
                    if element == " ":
                        if not hit_token:
                            num_end += 1
                        if hit_token:
                            break
                    elif element != " ":
                        hit_token = True

                correct_length = len(ent)
                print(
                    f"correcting currStart from {currStart} to {currStart + num_beginning}"
                )
                currStart = currStart + num_beginning

            else:
                if len(reftext[currStart:prevEnd]) > 2:
                    #     print("\nWEIRD entitiy")
                    #     print(
                    #         currStart,
                    #         prevEnd,
                    #         currType,
                    #         next_free_id_idx,
                    #         reftext[currStart:prevEnd],
                    #     )
                    #     print("~~~~~~~~~~~\n")
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
        if len(reftext[currStart:prevEnd]) > 2:
            # print("\nWEIRD entitiy")
            # print(
            #     currStart,
            #     prevEnd,
            #     currType,
            #     next_free_id_idx,
            #     reftext[currStart:prevEnd],
            # )
            # print("~~~~~~~~~~~\n")
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
    with open(text_file, "r", encoding="utf-8") as textf:

        text = textf.read()
        # text = unicodedata.normalize("NFKD", text)

    bio_str = ""

    for token_list, pred_list in zip(tokens, model_predictions):
        for token, pred in zip(token_list, pred_list):
            # normalize each token
            # bio_str += f"{unicodedata.normalize('NFKD', token)}\t{pred}\n"
            bio_str += f"{token}\t{pred}\n"

        bio_str += "\n"

    # print(f"bio str: {bio_str}")
    # print(f"text: {text}")
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
