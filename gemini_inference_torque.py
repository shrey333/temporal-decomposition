"""
Inference code using Gemini model for TORQUE dataset
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
DATASET_NAME = "torque"
MAX_SAMPLES = 100
DELAY = 15.0  # Delay between API calls in seconds
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

# Paths
TEST_DATA_PATH = Path("TORQUE/preprocess/dev.json")
OUTPUT_DIR = Path("results")
LOG_DIR = Path("./log")

# Gemini model setup
if not GOOGLE_API_KEY:
    raise ValueError("Please set GOOGLE_API_KEY in your environment variables")
genai.configure(api_key=GOOGLE_API_KEY)
MODEL = genai.GenerativeModel("gemini-pro")

# Example prompt template
PROMPT_TEMPLATE = """
You're an expert at identifying events from the context of a given question and providing a detailed explanation of why we chose those events for a given question. Give me a detailed explanation first, and then the answer starts with ##.

Definition of events:
When studying time, events were defined as actions/states triggered by verbs, adjectives, and nominals. One event corresponds to precisely one word. If there are no events, select None.


NOTE: Try to list all the events as they could have happened (re-arrange the given context). Then try to solve the question for each event from a re-arranged context and form the final answer.

Example 1:
question: What events started after Wei was left alone? (list down all events those are mentioned in context)
context: The New York Times said Wei had been under close watch at home, but was left briefly alone over the weekend. He then chose that moment to hang himself in the bathroom of his apartment, a colleague was quoted as saying. 
detailed explanation:
The events that started after Wei was left alone are as follows:
chose: Wei chose the moment to hang himself.
hang: Wei hung himself.
quoted: A colleague was quoted as saying something.
saying: The act of the colleague saying something.
said: The New York Times reported on the events.
These events are sequenced such that "chose" and "hang" are the immediate actions Wei took after being left alone. The "quoted" and "saying" refer to the colleague's statement, likely after Wei's actions. Finally, "said" by The New York Times is included as it represents reporting these events, which happens after all the preceding actions. Thus, all these events are part of the sequence that unfolded after Wei was left alone.
answer:
## Chose, hang, quoted, saying, said

Example 2:
question: What started after the New York Times said something? (list down all events those are mentioned in context)
context: The New York Times said Wei had been under close watch at home, but was left briefly alone over the weekend. He then chose that moment to hang himself in the bathroom of his apartment, a colleague was quoted as saying. 
detailed explanation:
Based on the provided context, the events described (Wei being under watch, being left alone, and choosing to hang himself) occurred before The New York Times reporting on them. The Times' statement is a retrospective account of these events. No subsequent events mentioned in the context started after the Times made their statement. Therefore, the correct answer is "None."
answer:
## None


Example 3:
question: What events have begun but has not finished? (list down all events those are mentioned in context)
context: Pakistan's defence ministry Sunday dismissed Indian reports of an alarming increase in cross border firing in the disputed Kashmir state. "It is a ploy to divert attention" from the turbulence in Indian-held Kashmir, a ministry official said.
detailed explanation:
Increase: The context mentions an "alarming increase in cross border firing," suggesting this is an ongoing process.
Firing: The act of firing is described as ongoing in the disputed region.
Divert: The phrase "ploy to divert attention" implies an ongoing effort to shift focus.
Turbulence: Describes the current state of unrest, which is ongoing.
Ploy: As a plan in action, it is considered an ongoing event.
Is: As a state of being, it is ongoing.
answer:
## increase, firing, divert, turbulence, is, ploy



Example 4:
question: What happened after local residents attacked UN vehicles?
context: Grandi said local residents attacked the UN vehicles to protest at the presence of Rwandan Hutu refugees in their area, saying the situation remained extremely tense although an uneasy calm had been restored. Witnesses said the incident was sparked by the killings of seven local villagers 20 kilometres (12 miles) from Kisangani, deaths blamed by locals on the refugees.
detailed explanation:
"said" (Grandi said): Grandi commented on the situation after the attack, indicating his statement occurred afterward.
"saying" (Grandi's saying): This refers to the content of Grandi's statement about the tense situation.
"remained" (situation remained tense): Describes the ongoing tension following the attack.
"had been restored" (uneasy calm had been restored): Indicates that a calm was established after the attack.
"said" (witnesses said): Witnesses provided information about the incident, which happened after the attack.
answer:
## said, saying, remained, restored, said


{input_text}
"""


class TorqueInferenceDataset(Dataset):
    """Dataset for TORQUE inference using Gemini"""

    def __init__(self, filepath: Path):
        with open(filepath) as f:
            examples = json.load(f)

        self.examples = [
            {
                "example_id": i,
                "input": f"question: {example['question']} context: {example['context']} answer: ",
                "output": ", ".join([x["mention"] for x in example["answers"]]),
            }
            for i, example in enumerate(examples)
        ]

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def parse_gemini_response(response_text: str) -> str:
    """Extract the answer after '##' from Gemini's response"""
    try:
        parts = response_text.split("##")
        if len(parts) > 1:
            return parts[-1].strip()
        return response_text.strip()
    except Exception as e:
        logging.error(f"Error parsing response: {str(e)}")
        return response_text.strip()


def generate_predictions(dataset: Dataset) -> List[Dict]:
    """Generate predictions using Gemini model"""
    results = []

    for example in tqdm(dataset[:MAX_SAMPLES]):
        try:
            # Format input and generate response
            prompt = PROMPT_TEMPLATE.format(input_text=example["input"])
            response = MODEL.generate_content(prompt)
            parsed_answer = parse_gemini_response(response.text)

            # Store results
            results.append(
                {
                    "prompt": prompt,
                    "actual_output": example["output"],
                    "parsed_answer": parsed_answer,
                    "gemini_output": response.text,
                    "is_correct": example["output"].lower() == parsed_answer.lower(),
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
    dataset = TorqueInferenceDataset(TEST_DATA_PATH)
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
