"""
Finetune/PEFT code for TORQUE dataset using Llama model
"""

import json
import logging
from pathlib import Path
from peft import (
    LoraConfig,
    TaskType,
    get_peft_model,
)
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup,
    GenerationConfig,
    PreTrainedTokenizer,
)
from tokenizers import processors

# Constants
DATASET_NAME = "torque"
MODEL_ID = "meta-llama/Llama-3.2-3B"
BATCH_SIZE = 8
NUM_EPOCHS = 10
PRECISION_TYPE = "bfloat16"
LEARNING_RATES = [1e-4, 1e-5, 1e-6]
LORA_DIMENSIONS = [16, 64]
LORA_ALPHAS = [16, 64]
LORA_DROPOUT = 0.1
DEFAULT_PAD_TOKEN = "[PAD]"

# Data paths
TRAIN_DATA_PATH = Path("TORQUE/preprocess/train_train.json")
DEV_DATA_PATH = Path("TORQUE/preprocess/train_dev.json")
LOG_DIR = Path("TORQUE/log")
OUTPUT_DIR = Path("TORQUE/output")


def _create_example(example: dict) -> tuple[str, str]:
    """Create input-output pair from example"""
    qst, ctx = example["question"], example["context"]
    input_text = f"question: {qst} context: {ctx} answer: "
    output_text = ", ".join([x["mention"] for x in example["answers"]])
    return input_text, output_text


class TorqueDataset(Dataset):
    """Dataset for TORQUE fine-tuning"""

    def __init__(
        self, filepath: Path, tokenizer: PreTrainedTokenizer, is_eval: bool = False
    ):
        with open(filepath) as f:
            examples = json.load(f)

        self.examples = [
            {"example_id": i, "input": input_text, "output": output_text}
            for i, example in enumerate(examples)
            for input_text, output_text in [_create_example(example)]
        ]
        self.tokenizer = tokenizer
        self.is_eval = is_eval

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        self.tokenizer.padding_side = "right"
        encoding = self.tokenizer(
            [(x["input"], x["output"]) for x in batch],
            padding=True,
            add_special_tokens=True,
            return_tensors="pt",
        )

        # Create labels for decoder-only model
        labels = [
            [
                (
                    input_ids[idx]
                    if seq_id == 1
                    or (idx > 0 and input_ids[idx] == self.tokenizer.eos_token_id)
                    else -100
                )
                for idx, seq_id in enumerate(encoding.sequence_ids(i))
            ]
            for i, input_ids in enumerate(encoding.input_ids)
        ]

        # Create eval inputs if needed
        eval_inputs = None
        if self.is_eval:
            self.tokenizer.padding_side = "left"
            eval_inputs = self.tokenizer(
                [x["input"] for x in batch],
                padding=True,
                add_special_tokens=True,
                return_tensors="pt",
            )

        return (
            encoding.input_ids,
            encoding.attention_mask,
            torch.LongTensor(labels),
            eval_inputs.input_ids if eval_inputs else None,
            eval_inputs.attention_mask if eval_inputs else None,
        )


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


def train_epoch(dataloader, model, optimizer, scheduler, device="cuda"):
    """Train for one epoch"""
    model.train()
    avg_loss = 0

    for batch in tqdm(dataloader):
        input_ids, attention_mask, labels = (
            x.to(device) if x is not None else None for x in batch[:3]
        )

        loss = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        ).loss

        avg_loss += loss.item()
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return avg_loss / len(dataloader)


def validate(dataloader, model, tokenizer, device="cuda"):
    """Validate model performance"""
    model.eval()
    total_loss = 0
    predictions = []

    with torch.no_grad():
        for batch in tqdm(dataloader):
            input_ids, attention_mask, labels, eval_ids, eval_mask = (
                x.to(device) if x is not None else None for x in batch
            )

            # Calculate loss
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            total_loss += outputs.loss.item()

            # Generate predictions
            if eval_ids is not None:
                generated = model.generate(
                    input_ids=eval_ids,
                    attention_mask=eval_mask,
                    max_new_tokens=64,
                )
                predictions.extend(
                    tokenizer.batch_decode(
                        generated.detach().cpu(), skip_special_tokens=True
                    )
                )

    # Calculate metrics
    exact_matches = []
    for example, pred in zip(dataloader.dataset, predictions):
        gold = [x.strip() for x in example["output"].split(",")]
        pred_clean = [x.strip() for x in pred.replace(example["input"], "").split(",")]
        exact_matches.append(
            len(gold) == len(pred_clean) and all(x in pred_clean for x in gold)
        )

    return (total_loss / len(dataloader), sum(exact_matches) / len(exact_matches))


def main():
    # Setup
    torch.manual_seed(7)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(7)

    tokenizer, num_new_tokens = setup_tokenizer(MODEL_ID)

    # Create datasets
    train_dataset = TorqueDataset(TRAIN_DATA_PATH, tokenizer)
    dev_dataset = TorqueDataset(DEV_DATA_PATH, tokenizer, is_eval=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=train_dataset.collate_fn,
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=BATCH_SIZE, collate_fn=dev_dataset.collate_fn
    )

    # Initialize model
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, device_map="auto"
    )
    if num_new_tokens:
        model.resize_token_embeddings(len(tokenizer))

    # Add LoRA
    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=LORA_DIMENSIONS[0],
            lora_alpha=LORA_ALPHAS[0],
            lora_dropout=LORA_DROPOUT,
        ),
    )

    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATES[0])
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(len(train_loader) * NUM_EPOCHS * 0.1),
        num_training_steps=len(train_loader) * NUM_EPOCHS,
    )

    # Create output directory
    output_dir = OUTPUT_DIR / DATASET_NAME / f"llama2-7b_{PRECISION_TYPE}_peft"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_score = 0
    for epoch in range(NUM_EPOCHS):
        train_loss = train_epoch(train_loader, model, optimizer, scheduler)
        val_loss, val_score = validate(dev_loader, model, tokenizer)

        if val_score > best_score:
            best_score = val_score
            model.save_pretrained(output_dir / f"best_model")

        # Log progress
        log = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_score": val_score,
            "best_score": best_score,
        }
        with open(output_dir / "training_log.json", "a") as f:
            json.dump(log, f)
            f.write("\n")


if __name__ == "__main__":
    main()
