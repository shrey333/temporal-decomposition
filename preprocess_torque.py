"""
Preprocessing script for TORQUE dataset
https://github.com/qiangning/TORQUE-dataset
"""

import json
import logging
from pathlib import Path
import random
from typing import List, Dict

# Constants
SEED = 7
TRAIN_DEV_SPLIT = 0.2

# Paths
INPUT_DIR = Path("TORQUE/data/TORQUE-dataset/")
OUTPUT_DIR = Path("TORQUE/preprocess/")
LOG_DIR = Path("./log/")


def _process_mentions(annotation: Dict, context: str) -> List:
    """Process annotated mentions"""
    mentions = []
    spans = annotation["spans"]
    indices = annotation["indices"]

    for span, offset in zip(spans, indices):
        start, end = offset.strip("()").split(",")
        start, end = int(start.strip()), int(end.strip())

        if context[start:end] != span:
            logging.warning(f"Index Mismatch: {context[start:end]} != {span}")
        else:
            mentions.append({"mention": span, "start": start, "end": end})

    return mentions


def preprocess_train(data: List) -> List:
    """Preprocess TORQUE train set"""
    examples = []

    for item in data:
        for passage in item["passages"]:
            context = passage["passage"]

            # Process annotated events
            events = []
            if len(passage["events"]) != 1:
                logging.warning(f"Unexpected format of events: {passage['events']}")
                continue
            else:
                events = _process_mentions(passage["events"][0]["answer"], context)

            # Process annotated question-answer pairs
            for qa in passage["question_answer_pairs"]:
                if qa["passageID"] != passage["events"][0]["passageID"]:
                    logging.warning(
                        f"passageID Mismatch: {qa['passageID']} "
                        f"!= {passage['events'][0]['passageID']}"
                    )
                    continue

                if not qa["isAnswered"]:
                    logging.warning(
                        f"The question, {qa['question_id']}, is not answered."
                    )
                else:
                    answers = _process_mentions(qa["answer"], context)

                    examples.append(
                        {
                            "context": context.strip(),
                            "events": events,
                            "answers": answers,
                            "question": qa["question"].strip(),
                            "passage_id": qa["passageID"],
                            "question_id": qa["question_id"],
                            "is_default": qa["is_default_question"],
                        }
                    )

    logging.info(f"#examples: {len(examples)}")
    return examples


def preprocess_dev(data: Dict) -> List:
    """Preprocess TORQUE dev set"""
    examples = []

    for doc_id, item in data.items():
        context = item["passage"]
        events = _process_mentions(item["events"]["answer"], context)

        for question, answers_indices in item["question_answer_pairs"].items():
            if answers_indices["passageID"] != item["events"]["passageID"]:
                logging.warning(
                    f"passageID Mismatch: {answers_indices['passageID']} "
                    f"!= {item['events']['passageID']}"
                )
                continue

            if answers_indices["validated_by"] != 3:
                logging.warning(
                    f"This question is annotated by "
                    f"{answers_indices['validated_by']}, not the default 3."
                )

            answers = _process_mentions(answers_indices["answer"], context)
            raw_answers_list = [
                _process_mentions(individual_answer, context)
                for individual_answer in answers_indices["individual_answers"]
            ]

            examples.append(
                {
                    "context": context.strip(),
                    "events": events,
                    "answers": answers,
                    "raw_answers_list": raw_answers_list,
                    "question": question.strip(),
                    "passage_id": doc_id,
                    "cluster_id": answers_indices["cluster_id"],
                    "is_default": answers_indices["is_default_question"],
                }
            )

    logging.info(f"#examples: {len(examples)}")
    return examples


def main():
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Process train data
    with open(INPUT_DIR / "train.json", "r") as f:
        data_train = json.load(f)
    examples_train = preprocess_train(data_train)

    with open(OUTPUT_DIR / "train.json", "w") as f:
        json.dump(examples_train, f, indent=4)
        f.write("\n")

    # Split train into train and dev
    random.seed(SEED)
    random.shuffle(examples_train)
    split_idx = int(len(examples_train) * TRAIN_DEV_SPLIT)
    examples_train_dev = examples_train[:split_idx]
    examples_train_train = examples_train[split_idx:]

    with open(OUTPUT_DIR / "train_train.json", "w") as f:
        json.dump(examples_train_train, f, indent=4)
        f.write("\n")

    with open(OUTPUT_DIR / "train_dev.json", "w") as f:
        json.dump(examples_train_dev, f, indent=4)
        f.write("\n")

    # Process dev data
    with open(INPUT_DIR / "dev.json", "r") as f:
        data_dev = json.load(f)
    examples_dev = preprocess_dev(data_dev)

    with open(OUTPUT_DIR / "dev.json", "w") as f:
        json.dump(examples_dev, f, indent=4)
        f.write("\n")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "preprocess.log"),
        ],
    )
    main()
