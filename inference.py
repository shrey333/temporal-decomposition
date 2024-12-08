"""
Inference code for TORQUE dataset using Llama model
"""

import json
import logging
from pathlib import Path
from peft import PeftModel
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    PreTrainedTokenizer,
)
from transformers.pipelines.pt_utils import KeyDataset
import torch
from torch.utils.data import Dataset

# Constants
DATASET_NAME = "torque"
MODEL_ID = "meta-llama/Llama-3.2-3B"
BATCH_SIZE = 8
PRECISION_TYPE = "bfloat16"
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.0
DEFAULT_PAD_TOKEN = "[PAD]"

# Paths
TEST_DATA_PATH = Path("TORQUE/preprocess/dev.json")
LOG_DIR = Path("./log")
OUTPUT_DIR = Path("TORQUE/output/benchmark")
SCORE_DIR = Path("TORQUE/output_score/benchmark")
PEFT_MODEL_PATH = Path(
    "/usr1/datasets/kimihiro/llm-for-event-temporal-ordering/models/torque/Llama-2-7b-hf_bfloat16_peft_generation/seed7_bs8_lr0.0001_dim16_alpha64_drop0.1"
)


def _create_example(example: dict) -> tuple[str, str]:
    """Create input-output pair from example"""
    qst, ctx = example["question"], example["context"]
    input_text = f"question: {qst} context: {ctx} answer: "
    output_text = ", ".join([x["mention"] for x in example["answers"]])
    return input_text, output_text


class TorqueInferenceDataset(Dataset):
    """Dataset for TORQUE inference"""

    def __init__(self, filepath: Path):
        with open(filepath) as f:
            examples = json.load(f)

        self.examples = [
            {"example_id": i, "input": input_text, "output": output_text}
            for i, example in enumerate(examples)
            for input_text, output_text in [_create_example(example)]
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def setup_tokenizer(tokenizer_id: str) -> tuple[PreTrainedTokenizer, int]:
    """Setup tokenizer for Llama model"""
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    num_new_tokens = 0

    if tokenizer.pad_token is None:
        num_new_tokens = tokenizer.add_special_tokens({"pad_token": DEFAULT_PAD_TOKEN})

        # Update post processor
        bos, eos = tokenizer.bos_token, tokenizer.eos_token
        single = f"{bos}:0 $A:0"
        pair = f"{bos}:0 $A:0 $B:1{eos}:1"
        tokenizer._tokenizer.post_processor = processors.TemplateProcessing(
            single=single,
            pair=pair,
            special_tokens=[
                (bos, tokenizer.bos_token_id),
                (eos, tokenizer.eos_token_id),
            ],
        )

    tokenizer.padding_side = "left"
    tokenizer.model_max_length = 4096
    tokenizer.truncation_side = "left"

    return tokenizer, num_new_tokens


def setup_model(tokenizer: PreTrainedTokenizer, num_new_tokens: int):
    """Setup model with PEFT"""
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )

    if num_new_tokens:
        model.resize_token_embeddings(len(tokenizer))

    model = PeftModel.from_pretrained(model, PEFT_MODEL_PATH)

    return model


def evaluate_predictions(dataset, predictions):
    """Evaluate model predictions"""
    results = []
    for example, pred in zip(dataset, predictions):
        gold = [x.strip() for x in example["output"].split(",")]
        pred_clean = [x.strip() for x in pred.replace(example["input"], "").split(",")]
        exact_match = len(gold) == len(pred_clean) and all(
            x in pred_clean for x in gold
        )
        results.append(
            {
                "input": example["input"],
                "gold": gold,
                "prediction": pred_clean,
                "exact_match": exact_match,
            }
        )

    accuracy = sum(r["exact_match"] for r in results) / len(results)
    return results, accuracy


def main():
    # Setup
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)

    # Initialize tokenizer and model
    tokenizer, num_new_tokens = setup_tokenizer(MODEL_ID)
    model = setup_model(tokenizer, num_new_tokens)

    # Create dataset
    dataset = TorqueInferenceDataset(TEST_DATA_PATH)
    logging.info(f"Loaded {len(dataset)} test examples")

    # Setup generator
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
    )

    # Generate predictions
    predictions = []
    for output in tqdm(
        generator(
            KeyDataset(dataset, "input"),
            batch_size=BATCH_SIZE,
        ),
        total=len(dataset),
    ):
        predictions.extend(output)

    # Evaluate predictions
    results, accuracy = evaluate_predictions(dataset, predictions)

    # Save results
    output_path = OUTPUT_DIR / DATASET_NAME / f"Llama-3.2-3B_{PRECISION_TYPE}_peft"
    output_path.mkdir(parents=True, exist_ok=True)

    output = {
        "config": {
            "model_id": MODEL_ID,
            "precision_type": PRECISION_TYPE,
            "batch_size": BATCH_SIZE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": TEMPERATURE,
            "peft_model_path": str(PEFT_MODEL_PATH),
        },
        "results": results,
        "accuracy": accuracy,
    }

    with open(output_path / "predictions.json", "w") as f:
        json.dump(output, f, indent=4)

    # Save scores
    score_path = SCORE_DIR / DATASET_NAME / f"Llama-3.2-3B_{PRECISION_TYPE}_peft"
    score_path.mkdir(parents=True, exist_ok=True)

    with open(score_path / "scores.json", "w") as f:
        json.dump({"accuracy": accuracy}, f, indent=4)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "inference.log"),
        ],
    )
    main()
