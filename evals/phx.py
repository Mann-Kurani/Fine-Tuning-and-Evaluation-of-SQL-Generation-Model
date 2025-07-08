# %%
import os
import json
import nest_asyncio
import evaluate

import phoenix as px
from phoenix.otel import register
from phoenix.experiments import run_experiment
from phoenix.experiments.types import Example
from opentelemetry.trace import StatusCode

from pydantic import BaseModel
from datasets import load_dataset

from transformers import AutoTokenizer, AutoModelForCausalLM
# from peft import PeftModel, LoraConfig, get_peft_model

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


# model = AutoModelForCausalLM.from_pretrained("/Users/mann.kurani.ctr/Desktop/Personal_Code/SQL Finetune/models/finetuned_final_tinyllama").to(device)



# %%
# Set environment variable for Phoenix
os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "http://localhost:6006"

# %%
# === Load dataset ===
ds = load_dataset("gretelai/synthetic_text_to_sql")
test_df = ds["test"].to_pandas().iloc[:500]

# %%
# === Initialize Phoenix session ===
def start_phoenix(project_name, df):
    session = px.launch_app()
    tracer_provider = register(protocol="http/protobuf", project_name=project_name)
    tracer = tracer_provider.get_tracer(__name__)

    client = px.Client()
    client.upload_dataset(
        dataframe=df,
        dataset_name="train_sql_dataset",
        input_keys=["sql_prompt", "sql_context"],
        output_keys=["sql"],
    )
    dataset = client.get_dataset(name="train_sql_dataset")
    return session, tracer, client, dataset

session, tracer, client, dataset = start_phoenix("finetuned_llama3_sql", test_df)

# %%
# === Output Schema for Phoenix tracing ===
class OutputSchema(BaseModel):
    sql: str

# %%
@tracer.tool()
def parse_model_output(text) -> OutputSchema:
    try:
        raw = json.loads(text)[0]
        return OutputSchema(**raw)
    except Exception:
        return OutputSchema(sql="ERROR GENERATING SQL QUERY")

# %%
# === Inference Logic ===
def model_experiment(model, tokenizer, sql_prompt, sql_context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that converts SQL prompts into SQL queries."},
        {"role": "user", "content": json.dumps({
            "sql_prompt": sql_prompt,
            "sql_context": sql_context
        })},
    ]

    with tracer.start_as_current_span("SQL Generation", openinference_span_kind="tool") as span:
        span.set_input(messages[1]["content"])



        # 1. Get the prompt string
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True  # Set this as needed
        )

        # 2. Tokenize the prompt string
        tokenized = tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to("mps")

        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,  # ✅ THIS FIXES THE WARNING
        max_new_tokens=256,
        pad_token_id=tokenizer.pad_token_id  # Optional, but avoids another warning
        )   

        # output_ids = model.generate(inputs, max_new_tokens=256)
        decoded = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        # print(decoded)
        generated = decoded.split("assistant")[-1].strip()
        # print(generated)

        span.set_output(generated)
        span.set_status(StatusCode.OK)

    return parse_model_output(generated)

# %%
# === Task Definition ===
@tracer.chain()
def task(example: Example) -> dict:
    sql_prompt = example.input["sql_prompt"]
    sql_context = example.input["sql_context"]
    output = model_experiment(model, tokenizer, sql_prompt, sql_context)
    return {"sql": output.sql}

# %%
# === Evaluators ===
smoother = SmoothingFunction().method4
bleu_metric = evaluate.load("bleu")

def sentence_bleu_score(reference: str, output: str) -> float:
    try:
        return sentence_bleu([reference["sql"]], output["sql"], smoothing_function=smoother)
    except:
        return 0.0

def exact_match_sql(reference: str, output: str) -> float:
    return 1.0 if output["sql"].strip() == reference["sql"].strip() else 0.0

def bleu_score(reference: str, output: str) -> float:
    try:
        result = bleu_metric.compute(predictions=[output["sql"]], references=[[reference["sql"]]])
        return result["bleu"]
    except:
        return 0.0

evaluators = [sentence_bleu_score, exact_match_sql, bleu_score]


# %%
# del model, tokenizer

# %%
# === Tokenizer and Model Setup ===
model_id = "meta-llama/Llama-3.2-1B-Instruct"
# finetuned_model_path = "/Users/mann.kurani.ctr/Desktop/Personal_Code/SQL Finetune/sql_generator_improved/checkpoint-2150"
finetuned_model_path = "/Users/mann.kurani.ctr/Desktop/Personal_Code/SQL Finetune/models/finetuned_final_3_tinyllama"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
tokenizer.pad_token = tokenizer.eos_token
# tokenizer.chat_template = '{{- bos_token }}\n{%- if custom_tools is defined %}\n    {%- set tools = custom_tools %}\n{%- endif %}\n{%- if not tools_in_user_message is defined %}\n    {%- set tools_in_user_message = true %}\n{%- endif %}\n{%- if not date_string is defined %}\n    {%- if strftime_now is defined %}\n        {%- set date_string = strftime_now("%d %b %Y") %}\n    {%- else %}\n        {%- set date_string = "26 Jul 2024" %}\n    {%- endif %}\n{%- endif %}\n{%- if not tools is defined %}\n    {%- set tools = none %}\n{%- endif %}\n\n{#- This block extracts the system message, so we can slot it into the right place. #}\n{%- if messages[0][\'role\'] == \'system\' %}\n    {%- set system_message = messages[0][\'content\']|trim %}\n    {%- set messages = messages[1:] %}\n{%- else %}\n    {%- set system_message = "" %}\n{%- endif %}\n\n{#- System message #}\n{{- "<|start_header_id|>system<|end_header_id|>\\n\\n" }}\n{%- if tools is not none %}\n    {{- "Environment: ipython\\n" }}\n{%- endif %}\n{%- if tools is not none and not tools_in_user_message %}\n    {{- "You have access to the following functions. To call a function, please respond with JSON for a function call." }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n{%- endif %}\n{{- system_message }}\n{{- "<|eot_id|>" }}\n\n{#- Custom tools are passed in a user message with some extra guidance #}\n{%- if tools_in_user_message and not tools is none %}\n    {#- Extract the first user message so we can plug it in here #}\n    {%- if messages | length != 0 %}\n        {%- set first_user_message = messages[0][\'content\']|trim %}\n        {%- set messages = messages[1:] %}\n    {%- else %}\n        {{- raise_exception("Cannot put tools in the first user message when there\'s no first user message!") }}\n{%- endif %}\n    {{- \'<|start_header_id|>user<|end_header_id|>\\n\\n\' -}}\n    {{- "Given the following functions, please respond with a JSON for a function call " }}\n    {{- "with its proper arguments that best answers the given prompt.\\n\\n" }}\n    {{- \'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}.\' }}\n    {{- "Do not use variables.\\n\\n" }}\n    {%- for t in tools %}\n        {{- t | tojson(indent=4) }}\n        {{- "\\n\\n" }}\n    {%- endfor %}\n    {{- first_user_message + "<|eot_id|>"}}\n{%- endif %}\n\n{%- for message in messages %}\n    {%- if not (message.role == \'ipython\' or message.role == \'tool\' or \'tool_calls\' in message) %}\n        {{- \'<|start_header_id|>\' + message[\'role\'] + \'<|end_header_id|>\\n\\n\'+ message[\'content\'] | trim + \'<|eot_id|>\' }}\n    {%- elif \'tool_calls\' in message %}\n        {%- if not message.tool_calls|length == 1 %}\n            {{- raise_exception("This model only supports single tool-calls at once!") }}\n        {%- endif %}\n        {%- set tool_call = message.tool_calls[0].function %}\n        {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' -}}\n        {{- \'{"name": "\' + tool_call.name + \'", \' }}\n        {{- \'"parameters": \' }}\n        {{- tool_call.arguments | tojson }}\n        {{- "}" }}\n        {{- "<|eot_id|>" }}\n    {%- elif message.role == "tool" or message.role == "ipython" %}\n        {{- "<|start_header_id|>ipython<|end_header_id|>\\n\\n" }}\n        {%- if message.content is mapping or message.content is iterable %}\n            {{- message.content | tojson }}\n        {%- else %}\n            {{- message.content }}\n        {%- endif %}\n        {{- "<|eot_id|>" }}\n    {%- endif %}\n{%- endfor %}\n{%- if add_generation_prompt %}\n    {{- \'<|start_header_id|>assistant<|end_header_id|>\\n\\n\' }}\n{%- endif %}\n'

model = AutoModelForCausalLM.from_pretrained(finetuned_model_path).to("mps")
# model = PeftModel.from_pretrained(model, "/Users/mann.kurani.ctr/Desktop/Personal_Code/SQL Finetune/sql_generator_improved/checkpoint-800")


# %%
# === Run Experiment ===
nest_asyncio.apply()
experiment = run_experiment(
    dataset=dataset,
    task=task,
    experiment_name="finetuned_3_llama3_sql_eval",
    evaluators=evaluators
)

# %%
# checkpoint-800
# Experiment Summary (07/01/25 11:49 PM +0530)
# --------------------------------------------
#              evaluator   n  n_scores  avg_score
# 0           bleu_score  50        50   0.611605
# 1      exact_match_sql  50        50   0.340000
# 2  sentence_bleu_score  50        50   0.749032

# %%



