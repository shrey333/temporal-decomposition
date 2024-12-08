"""
Inference code using Gemini model for MATRES dataset
"""

import json
import logging
import os
from pathlib import Path
import time
from typing import List, Dict
import google.generativeai as genai
from tqdm import tqdm
from torch.utils.data import Dataset

# Constants
DATASET_NAME = "matres"
MAX_SAMPLES = 50
DELAY = 30.0  # Delay between API calls in seconds
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
TEST_DATA_PATH = Path("MATRES/preprocess/test.json")
OUTPUT_DIR = Path("MATRES/results")
LOG_DIR = Path("./log")

# Gemini model setup
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your environment variables")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel("gemini-pro")

# Example prompt template
PROMPT_TEMPLATE = """
I'm working on the timeline construction task using LLM and need your assistance in this task. I want to decompose the context which is my input. 

The context is a paragraph and has tags [e1] and [e2] indicating the events 1 and 2. As a first task we need to identify the events 1 and 2. 

Then, once the tasks are identified, Let the start of event 1 be considered as t1 and start of event 2 be considered as t2.
Label set for t1 start vs. t2 start is "before", "after", "equal" and "vague". 
We ask two questions: 
Q1 = Is it possible that t1 start is before t2 start? 
Q2 = Is it possible that t2 start is before t1 start? 
Let the answers be A1 and A2. Then we have a one-to-one mapping as follows: 
A1 = A2 = yes → vague,
A1 = A2 = no → equal
A1 = yes, A2 = no → before, and
A1 = no, A2 = yes → after.

for example:
context: Mr. Obama [e1]said[/e1] later at a news conference in Amman that he had [e2]spoken[/e2] to both leaders over the past two years about how it was in the interests of both countries to restore normal relations.

the identified events 1 and 2 are "said" and "spoken" respectively.

According to our Q1 and Q2. 
Q1: Is it possible that "said" start is before "spoken" start?
Q2: Is it possible that "spoken" start is before "said" start?

Answers are: A1: no and A2: yes. 
hence the answer is after (basis above rules.)
answer: after

Now, identify the events 1 and 2 and find the relation between the identified events in the below input and don't give me any other details. Just give me the relation between the identified events as 'before', 'after', 'equal' or 'vague': 

{input_text}
"""


class MatresInferenceDataset(Dataset):
    """Dataset for MATRES inference using Gemini"""

    def __init__(self, filepath: Path):
        with open(filepath) as f:
            examples = json.load(f)

        self.examples = [
            {
                "example_id": i,
                "input": example["context"],
                "output": example["relation"],
            }
            for i, example in enumerate(examples)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def generate_predictions(dataset: Dataset) -> List[Dict]:
    """Generate predictions using Gemini model"""
    results = []

    for example in tqdm(dataset[:MAX_SAMPLES]):
        try:
            # Format input and generate response
            prompt = PROMPT_TEMPLATE.format(input_text=example["input"])
            response = MODEL.generate_content(prompt)
            prediction = response.text.strip().lower()

            # Store results
            results.append(
                {
                    "prompt": prompt,
                    "actual_output": example["output"],
                    "gemini_output": response.text,
                    "is_correct": example["output"].lower() == prediction,
                }
            )

            time.sleep(DELAY)  # Rate limiting

        except Exception as e:
            logging.error(f"Error processing example: {str(e)}")
            continue

    return results


def main():
    # Setup directories
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    # Load dataset
    dataset = MatresInferenceDataset(TEST_DATA_PATH)
    logging.info(f"Loaded {len(dataset)} examples")

    # Generate predictions
    results = generate_predictions(dataset)

    # Calculate accuracy
    accuracy = sum(r["is_correct"] for r in results) / len(results)

    # Save results
    output = {
        "config": {
            "model": "gemini-pro",
            "max_samples": MAX_SAMPLES,
            "delay": DELAY,
        },
        "results": results,
        "accuracy": accuracy,
    }

    output_path = OUTPUT_DIR / f"gemini_results_{DATASET_NAME}.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)

    logging.info(f"Processed {len(results)} examples with accuracy {accuracy:.4f}")
    logging.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s:%(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(LOG_DIR / "gemini_inference.log"),
        ],
    )
    main()
