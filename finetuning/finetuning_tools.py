import torch
import torch.nn as nn
from datasets import load_dataset, Dataset, DatasetDict
import pandas as pd
import json


def dataset_trainer(tokenizer, n=1000) -> DatasetDict:
    """Prepares and splits the dataset for training."""
    df = dataset_creation(n)
    prompts, target_responses = get_prompts_and_targets(df.to_pandas())

    data = generate_input_output_pairs(
        tokenizer=tokenizer,
        prompts=prompts,
        target_responses=target_responses
    )

    # Convert to DataFrame
    data_df = pd.DataFrame({key: value.tolist() for key, value in data.items()})
    dataset = Dataset.from_pandas(data_df)

    # Shuffle and split
    return dataset.shuffle(seed=42).train_test_split(test_size=0.2)


def dataset_creation(n: int) -> Dataset:
    """Loads and trims the text-to-SQL dataset."""
    dataset = load_dataset("gretelai/synthetic_text_to_sql")
    columns_to_keep = ['sql_prompt', 'sql_context', 'sql']

    # Filter only required columns
    dataset = dataset["train"].remove_columns(
        [col for col in dataset["train"].column_names if col not in columns_to_keep]
    )

    return dataset.select(range(n))


def get_prompts_and_targets(train_df: pd.DataFrame):
    """Constructs prompts and targets from the dataset."""
    prompts = []
    targets = []

    for _, row in train_df.iterrows():
        sql_prompt = row["sql_prompt"]
        sql_context = row["sql_context"]
        sql = row["sql"]

        # Format prompt as chat template
        chat_prompt = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant that converts natural language prompts into a single SQL query only.\n"
                    "You will be given:\n"
                    "- `sql_prompt`: A natural language instruction.\n"
                    "- `sql_context`: The database schema.\n\n"
                    "Generate a valid single SQL query. Respond only in JSON format like:\n"
                    "[{\"sql\": \"<your SQL query>\"}]"
                )
            },
            {
                "role": "user",
                "content": json.dumps({
                    "sql_prompt": sql_prompt,
                    "sql_context": sql_context
                })
            },
            {
                "role": "assistant",
                "content": ""  # Let the model generate this
            }
        ]

        # Target SQL wrapped in the required JSON format
        target_response = f'[{{"sql": "{sql}"}}]'
        prompts.append(chat_prompt)
        targets.append(target_response)

    return prompts, targets


def generate_input_output_pairs(tokenizer, prompts, target_responses):
    """Tokenizes input-output pairs with correct masking."""
    # Step 1: Create full input sequences with SQL appended
    chat_templates = tokenizer.apply_chat_template(
        prompts,
        tokenize=False,
        add_generation_prompt=False
    )

    full_texts = [
        chat + target + tokenizer.eos_token
        for chat, target in zip(chat_templates, target_responses)
    ]

    # Tokenize full input (prompt + SQL)
    encoding = tokenizer(
        full_texts,
        return_tensors="pt",
        padding=True,
        truncation=True
    )

    input_ids = encoding.input_ids
    attention_mask = encoding.attention_mask

    # Step 2: Mask prompt tokens in labels
    prompt_lens = [
        len(tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True))
        for prompt in prompts
    ]


    labels = input_ids.clone()
    for i, prompt_len in enumerate(prompt_lens):
        labels[i, :prompt_len] = -100  # Ignore prompt in loss
    labels[labels == tokenizer.pad_token_id] = -100  # Ignore padding

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


def calculate_loss(logits, labels):
    """Computes masked cross-entropy loss."""
    loss_fn = nn.CrossEntropyLoss(reduction="none")
    return loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))



def create_training_prompt(sql_prompt, sql_context):
    # Define the chat template
    training_prompt = [
        {
            "role": "system",
            "content": (
                "You are a helpful assistant that converts sql prompt and sql context into a SQL query."
            )
        },
        {
            "role": "user",
            "content": json.dumps({
                "sql_prompt": sql_prompt,
                "sql_context": sql_context
            })
        },
        {
            "role": "assistant",
            "content": ""  # Leave blank to let model generate fully
        }
    ]
    return training_prompt

def tokenize_fn(batch, tokenizer):
    # Step 1: Create prompt strings
    prompt_texts = [
        tokenizer.apply_chat_template(
            create_training_prompt(sql_prompt, sql_context),
            tokenize=False,
            add_generation_prompt=False,
            padding = "longest",
            truncation=True
        )
        for sql_prompt, sql_context in zip(batch["sql_prompt"], batch["sql_context"])
    ]

    # Step 2: Create target label strings
    label_texts = [
        f'[{{"sql": "{sql}"}}]{tokenizer.eos_token}'
        for sql in batch["sql"]
    ]

    # Step 3: Combine prompt + label
    full_texts = [
        prompt + label
        for prompt, label in zip(prompt_texts, label_texts)
    ]

    # Step 4: Tokenize full inputs
    full_encodings = tokenizer(
        full_texts,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    input_ids = full_encodings.input_ids
    attention_mask = full_encodings.attention_mask

    # Step 5: Compute label lengths
    label_lens = [
        len(tokenizer(label, add_special_tokens=False)["input_ids"])
        for label in label_texts
    ]

    # Step 6: Create labels by masking prompt tokens
    labels = input_ids.clone()
    for i, label_len in enumerate(label_lens):
        labels[i, :-label_len] = -100  # Mask everything before label

    # Step 7: Also mask padding
    labels[labels == tokenizer.pad_token_id] = -100

    # Replace the last non-padding token with the EOS token for each sequence in labels
    for i, label_seq in enumerate(labels):
        # Find the index of the last non-padding token in the sequence
        last_token_index = torch.max((label_seq != tokenizer.pad_token_id).nonzero(as_tuple=False))
        # Replace the last non-padding token with the EOS token ID
        labels[i, last_token_index] = tokenizer.eos_token_id
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }






