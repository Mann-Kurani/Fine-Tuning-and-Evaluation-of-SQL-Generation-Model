
import json
import numpy as np
import evaluate
from transformers import AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model
# import torch.nn as nn
from datasets import load_dataset, Dataset
import torch
import time
# from peft import PeftModel


device = "mps"

# model_id = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
model_id = "meta-llama/Llama-3.2-1B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
if not tokenizer.pad_token:
    tokenizer.pad_token = tokenizer.eos_token
# tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'

# model = AutoModelForCausalLM.from_pretrained("/Users/mann.kurani.ctr/Desktop/Personal_Code/SQL Finetune/models/finetuned_final_tinyllama").to(device)
model = AutoModelForCausalLM.from_pretrained("/Users/mann.kurani.ctr/Desktop/Personal_Code/SQL Finetune/models/finetuned_final_tinyllama").to(device)
# model = PeftModel.from_pretrained(model, "/Users/mann.kurani.ctr/Desktop/Personal_Code/SQL Finetune/models/finetuned_final_tinyllama").to(device)

lora_config = LoraConfig(
    task_type="CAUSAL_LM",
    r = 16,
    lora_alpha=32,
    lora_dropout=0.1,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()


def dataset_creation(n: int) -> Dataset:
    """Loads and trims the text-to-SQL dataset."""
    dataset = load_dataset("gretelai/synthetic_text_to_sql")
    columns_to_keep = ['sql_prompt', 'sql_context', 'sql']

    # Filter only required columns
    dataset = dataset["train"].remove_columns(
        [col for col in dataset["train"].column_names if col not in columns_to_keep]
    )

    return dataset.select(range(5000, n))


def tokenize_fn(batch):
    prompts = []
    sql_outputs = []

    for sql_prompt, sql_context, sql in zip(batch["sql_prompt"], batch["sql_context"], batch["sql"]):
        # Build the prompt using chat format
        prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": "You are a helpful assistant that converts SQL prompts and context into SQL queries. Do not give any additional comments or anything other than the sql query itself."},
                {"role": "user", "content": json.dumps({
                    "sql_prompt": sql_prompt,
                    "sql_context": sql_context
                })}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        prompts.append(prompt)

        # Format SQL output with EOS
        sql_output = f'[{{"sql": "{sql.strip()}"}}]{tokenizer.eos_token}'
        sql_outputs.append(sql_output)

    # Full sequence = prompt + sql_output
    full_texts = [p + o for p, o in zip(prompts, sql_outputs)]

    # Tokenize full sequences
    full_encodings = tokenizer(
        full_texts,
        truncation=True,
        max_length=512,
        padding="max_length",
        return_tensors="pt"
    )

    input_ids = full_encodings["input_ids"]
    attention_mask = full_encodings["attention_mask"]

    # Get prompt lengths to mask label tokens for prompts
    prompt_lengths = [
        len(tokenizer(p, truncation=True, max_length=512)["input_ids"])
        for p in prompts
    ]

    labels = input_ids.clone()
    for i, prompt_len in enumerate(prompt_lengths):
        labels[i, :prompt_len] = -100  # Mask prompt tokens

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels
    }


df = dataset_creation(10000).train_test_split(0.01)


train_df = df["train"]
eval_df = df["test"]


train_df = train_df.map(tokenize_fn, batched=True)
eval_df = eval_df.map(tokenize_fn, batched=True)


# Load BLEU metric once
bleu_metric = evaluate.load("bleu")

def compute_metrics(eval_pred, compute_result: bool = False):
    logits, labels = eval_pred

    logits = logits.cpu().numpy()
    labels = labels.to("cpu")

    # Pad labels and predictions to same shape if needed
    # if logits.shape[-1] != labels.shape[-1]:
    #     labels = labels[:, :logits.shape[1]]

    # Predicted token IDs (greedy)
    predictions = np.argmax(logits, axis=-1)

    # Replace -100 in labels with pad_token_id before decoding
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)

    # Decode
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Clean whitespace
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [label.strip() for label in decoded_labels]

    # print(decoded_preds)
    # print(decoded_labels)

    # BLEU requires list-of-references format
    result = bleu_metric.compute(
        predictions=decoded_preds,
        references=[[label] for label in decoded_labels]
    )

    # Exact Match
    exact_match = np.mean([
        int(pred == label)
        for pred, label in zip(decoded_preds, decoded_labels)
    ])

    # print({
    #     "bleu": result,          # BLEU score %
    #     "exact_match": exact_match       # EM score %
    # })

    result =  {
        "bleu": round(result["bleu"] * 100, 2),          # BLEU score %
        "exact_match": round(exact_match * 100, 2)       # EM score %
    }

    if compute_result:
        return result
    return result 

training_args = TrainingArguments(
    output_dir="checkpoints",

    # Basic training setup
    num_train_epochs=10,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    gradient_checkpointing=False,
    # # Evaluation, logging, and saving
    # eval_strategy="epoch",
    # logging_strategy="steps",
    # logging_steps=10,
    # save_strategy="epoch",
    # save_total_limit=2,
    
    eval_strategy="steps",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="steps",
    save_total_limit=10,
    save_steps = 50,
    eval_steps=50,
    batch_eval_metrics=True,

    # Learning parameters
    learning_rate=2e-4,
    warmup_steps=50,
    weight_decay=0.01,
    label_names=["labels"],

    # Device & precision
    fp16=False,
    bf16=False,
    dataloader_pin_memory=False,

    # Meta & checkpointing
    seed=42,
    report_to="tensorboard",
    logging_dir="./logs",
    load_best_model_at_end=True,
    # metric_for_best_model="exact_match",
    # greater_is_better=True,
)


data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_df,
    eval_dataset=eval_df,
    compute_metrics=compute_metrics,
    data_collator=data_collator,
    processing_class = tokenizer,
)

max_retries = 55
final_model_dir = "/Users/mann.kurani.ctr/Desktop/Personal_Code/SQL Finetune/models/finetuned_final_3_adaptor_tinyllama"
# trainer.evaluate()
for attempt in range(max_retries):
    try:
        # if attempt == 0:
        #     # Start fresh
        #     trainer.train()
        # else:
        #     # Resume from checkpoint
        #     trainer.train(resume_from_checkpoint=True)
        trainer.train(resume_from_checkpoint=True)
        trainer.model.save_pretrained(final_model_dir)
        print("Training complete. Model saved.")
        break

    except RuntimeError as e:
        print(f"[Attempt {attempt + 1}] RuntimeError during training: {e}")
        print("Retrying after short delay...")
        time.sleep(10)
        torch.mps.empty_cache()
else:
    print("Training failed after maximum retries.")



# do train.evaluate with printing the custom metrics, what is the issue? why is it showing 0?







