"""
Preprocessing script for MATRES dataset
"""

import json
import logging
from pathlib import Path
from typing import Any
from bs4 import BeautifulSoup
from collections import defaultdict
import random
import re
import spacy

# Constants
SEED = 7
TRAIN_DEV_SPLIT = 0.2
SPACY_MODEL = "en_core_web_sm"

# Paths
TIMEBANK_DIR = Path(
    "MATRES/StructTempRel-EMNLP17/data/TempEval3/Training/TBAQ-cleaned/TimeBank/"
)
AQUAINT_DIR = Path(
    "MATRES/StructTempRel-EMNLP17/data/TempEval3/Training/TBAQ-cleaned/AQUAINT/"
)
PLATINUM_DIR = Path(
    "MATRES/StructTempRel-EMNLP17/data/TempEval3/Evaluation/te3-platinum/"
)
ANNOTATION_DIR = Path("MATRES/data/")
OUTPUT_DIR = Path("MATRES/preprocess/")
LOG_DIR = Path("./log/")

# Source document paths
SPLIT_DIRS = {
    "timebank": TIMEBANK_DIR,
    "aquaint": AQUAINT_DIR,
    "platinum": PLATINUM_DIR,
}


def parse_annotation(dirpath: Path) -> dict[str, Any]:
    """Parse MATRES annotations"""
    annotation = {}
    for filepath in dirpath.glob("*.txt"):
        with open(filepath, "r") as f:
            lines = f.readlines()

        filename2annotation = defaultdict(list)
        for line in lines:
            splits = line.strip().split("\t")
            filename2annotation[splits[0]].append(
                {"eiid1": splits[3], "eiid2": splits[4], "relation": splits[5]}
            )
        annotation[filepath.stem] = filename2annotation

    return annotation


def parse_text(text: str, pipeline) -> dict[str, Any]:
    """Parse text using spaCy"""
    sentences = []
    for paragraph in [x for x in text.split("\n") if x]:
        for sentence in pipeline(paragraph).sents:
            sentence = sentence.text.strip()
            if not sentence:
                continue
            sentences.append(
                {
                    "text": sentence,
                    "start": text.index(sentence),
                    "end": text.index(sentence) + len(sentence),
                }
            )
    return sentences


def parse_tags(parsed_data) -> dict[str, Any]:
    """Parse XML tags from data"""
    text_with_tags = str(parsed_data.find("TEXT"))
    text = parsed_data.find("TEXT").text
    tags = (
        parsed_data.find("TEXT").find_all("EVENT")
        + parsed_data.find("TEXT").find_all("TIMEX3")
        + parsed_data.find("TEXT").find_all("SIGNAL")
    )

    # Sort tags
    tags_positions = []
    for tag in tags:
        if str(tag) in text_with_tags:
            tags_positions.append(
                {
                    "eid": tag.get("eid"),
                    "event_text": tag.text,
                    "tag_offset": text_with_tags.index(str(tag)),
                    "tag_length": len(str(tag)),
                }
            )
        else:
            logging.error(f"{tag} not found in text.")

    sorted_tags = sorted(tags_positions, key=lambda x: x["tag_offset"])

    # Map event IDs
    eid2eiid = {}
    for tag in parsed_data.find_all("MAKEINSTANCE"):
        eid = tag.get("eventID")
        eiid = tag.get("eiid")
        eid2eiid[eid] = eiid

    # Process events
    and_markers = [x.start() for x in re.finditer("\&", text_with_tags)]
    eiid2events = {}
    char_count_tags = len("<TEXT>")
    modifier = 0

    for item in sorted_tags:
        eid, event_text = item["eid"], item["event_text"]
        tag_offset, tag_length = item["tag_offset"], item["tag_length"]

        while and_markers and and_markers[0] < tag_offset:
            modifier += 4
            and_markers.pop(0)

        if eid and eid in eid2eiid:
            start = tag_offset - char_count_tags - modifier
            end = start + len(event_text)
            eiid = eid2eiid[eid].replace("ei", "")

            if event_text.startswith(" "):
                event_text = event_text[1:]
                start += 1
                char_count_tags -= 1

            eiid2events[eiid] = {"mention": event_text, "start": start, "end": end}

            if event_text != text[start:end]:
                logging.error(
                    f"[offset mismatch] {eid}: {event_text} != {text[start:end]}"
                )

        char_count_tags += tag_length - len(event_text)

    return eiid2events


def parse(filepath: Path, pipeline) -> tuple[list, dict[str, Any]]:
    """Parse a single document"""
    with open(filepath, "r") as f:
        raw_data = f.read()

    parsed_data = BeautifulSoup(raw_data, "xml")
    sentences = parse_text(parsed_data.find("TEXT").text, pipeline)
    eiid2events = parse_tags(parsed_data)

    return sentences, eiid2events


def find_source_sentence(sentences: list, event: dict[str, Any]) -> int:
    """Find which sentence contains the event"""
    for sent_id, sentence in enumerate(sentences):
        if sentence["start"] <= event["start"] and event["end"] <= sentence["end"]:
            return sent_id
    return -1


def update_offset(event: dict[str, Any], sentence: dict[str, Any]) -> dict[str, Any]:
    """Update event offsets relative to sentence"""
    start = event["start"] - sentence["start"]
    end = event["end"] - sentence["start"]

    if event["mention"] != sentence["text"][start:end]:
        logging.error(
            f"text mismatch: {event['mention']} != {sentence['text'][start:end]}"
        )

    return {
        "mention": event["mention"],
        "start": start,
        "end": end,
    }


def make_example(
    sentences: list, eiid2events: dict[str, Any], raw_annotation: dict[str, Any]
) -> dict[str, Any]:
    """Create a single example from annotation"""
    if (
        raw_annotation["eiid1"] not in eiid2events
        or raw_annotation["eiid2"] not in eiid2events
    ):
        logging.error(
            f"ei{raw_annotation['eiid1']} or ei{raw_annotation['eiid2']} "
            f"not identified in the tml file."
        )
        return None

    event1 = eiid2events[raw_annotation["eiid1"]]
    event2 = eiid2events[raw_annotation["eiid2"]]
    sent_id_event1 = find_source_sentence(sentences, event1)
    sent_id_event2 = find_source_sentence(sentences, event2)

    if sent_id_event1 == -1 or sent_id_event2 == -1:
        logging.error(
            f"[corresponding sentence not found] "
            f"e1: {sent_id_event1}, e2: {sent_id_event2}"
        )
        return None

    # Create context and update offsets
    sent_id_begin = min(sent_id_event1, sent_id_event2)
    sent_id_end = max(sent_id_event1, sent_id_event2)

    context = ""
    for sentence in sentences[sent_id_begin:sent_id_end]:
        context += f"{sentence['text']} "
    context += sentences[sent_id_end]["text"]

    event = {
        "arg1": update_offset(event1, sentences[sent_id_event1]),
        "arg2": update_offset(event2, sentences[sent_id_event2]),
    }

    # Adjust offsets for later event
    if sent_id_event1 < sent_id_event2:
        event["arg2"]["start"] += (
            len(" ".join(s["text"] for s in sentences[sent_id_event1:sent_id_event2]))
            + 1
        )
        event["arg2"]["end"] += (
            len(" ".join(s["text"] for s in sentences[sent_id_event1:sent_id_event2]))
            + 1
        )
    elif sent_id_event1 > sent_id_event2:
        event["arg1"]["start"] += (
            len(" ".join(s["text"] for s in sentences[sent_id_event2:sent_id_event1]))
            + 1
        )
        event["arg1"]["end"] += (
            len(" ".join(s["text"] for s in sentences[sent_id_event2:sent_id_event1]))
            + 1
        )

    return {
        "context": context,
        "arg1": event["arg1"],
        "arg2": event["arg2"],
        "relation": raw_annotation["relation"],
    }


def main():
    # Setup
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    pipeline = spacy.load(SPACY_MODEL)
    split2annotation = parse_annotation(ANNOTATION_DIR)

    # Process all splits
    examples = []
    for split_name, filename2annotation in split2annotation.items():
        logging.info(f"Processing {split_name} split")

        for filename, raw_annotations in filename2annotation.items():
            logging.info(f"Processing file: {filename}")

            sentences, eiid2events = parse(
                SPLIT_DIRS[split_name] / f"{filename}.tml", pipeline
            )

            for raw_annotation in raw_annotations:
                if example := make_example(sentences, eiid2events, raw_annotation):
                    examples.append(
                        {"split_name": split_name, "filename": filename, **example}
                    )

    # Split data
    examples_train_dev = [ex for ex in examples if ex["split_name"] != "platinum"]
    examples_test = [ex for ex in examples if ex["split_name"] == "platinum"]

    random.seed(SEED)
    random.shuffle(examples_train_dev)
    split_idx = int(len(examples_train_dev) * TRAIN_DEV_SPLIT)

    examples_dev = examples_train_dev[:split_idx]
    examples_train = examples_train_dev[split_idx:]

    # Save splits
    for name, data in [
        ("train", examples_train),
        ("dev", examples_dev),
        ("test", examples_test),
    ]:
        with open(OUTPUT_DIR / f"{name}.json", "w") as f:
            json.dump(data, f, indent=4)
            f.write("\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "preprocess_matres.log"),
        ],
    )
    main()
