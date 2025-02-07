# Copyright 2024 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import random
from dataclasses import dataclass, field
from typing import Any, Optional, Union

import evaluate
import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    set_seed,
)
from transformers.utils import PaddingStrategy
import math
from scipy.special import expit


# Define and parse arguments.
@dataclass
class ScriptArguments:
    """
    These arguments vary depending on how many GPUs you have, what their capacity and features are, and what size model you want to train.
    """

    local_rank: Optional[int] = field(default=-1, metadata={"help": "Used for multi-gpu"})
    resume_from_checkpoint: Optional[bool] = field(
        default=False, metadata={"help": "If you want to resume training where it left off."},
    )
    deepspeed: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to deepspeed config if using deepspeed. You may need this if the model that you want to train doesn't fit on a single GPU."
        },
    )
    per_device_train_batch_size: Optional[int] = field(default=2)
    per_device_eval_batch_size: Optional[int] = field(default=1)
    gradient_accumulation_steps: Optional[int] = field(default=1)
    learning_rate: Optional[float] = field(default=3e-5) # default was 2e-5
    weight_decay: Optional[float] = field(default=0.001)
    model_name: Optional[str] = field(
        default="meta-llama/Meta-Llama-3-8B",
        metadata={
            "help": "The model that you want to train from the Hugging Face hub. E.g. gpt2, gpt2-xl, bert, etc."
        },
    )
    root_dir: Optional[str] = field(
        default="/data1/Arash",
        metadata={
            "help": "Root directory where the input dataset, base model, and tokenizer are located and model checkpoints will be saved."
        }
    )
    train_dataset: Optional[str] = field(
        default="sum_rew",
        metadata={"help": "Type of preference dataset used for training {'sum_rew', 'rnd_rew', 'avg_rew'}"}
    )
    train_obj: Optional[str] = field(
        default="dpo",
        metadata={"help": "Type of preference learning objective {'dpo' for alignment, 'rew' for reward learning}"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "The tokenizer for your model, if left empty will use the default for your model"}
    )
    bf16: Optional[bool] = field(
        default=True,
        metadata={
            "help": "This essentially cuts the training time in half if you want to sacrifice a little precision and have a supported GPU."
        },
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "The number of training epochs for the reward model."},
    )
    gradient_checkpointing: Optional[bool] = field(
        default=False, metadata={"help": "Enables gradient checkpointing."},
    )
    optim: Optional[str] = field(
        default="adamw_hf", metadata={"help": "The optimizer to use."},
    )
    lr_scheduler_type: Optional[str] = field(
        default="linear", metadata={"help": "The lr scheduler"},
    )
    max_length: Optional[int] = field(default=512)
    eval_first_step: Optional[bool] = field(
        default=False, metadata={"help": "Whether to run eval after the first step"},
    )
    seed: Optional[int] = field(
        default=0, metadata={"help": "Random seed that will be set at the beginning of training."}
    )
    num_user_types: Optional[int] = field(
        default=3, metadata={"help": "Number of distinct user types with diverse preferences"}
    )


parser = HfArgumentParser(ScriptArguments)
script_args = parser.parse_args_into_dataclasses()[0]
set_seed(script_args.seed)
# Load the hh-rlhf dataset
train_dataset = load_dataset(path=f"{script_args.root_dir}/hh-rlhf", split="train", verification_mode="no_checks")
eval_dataset = load_dataset(path=f"{script_args.root_dir}/hh-rlhf", split="test", verification_mode="no_checks")
# Define the training args. Needs to be done before the model is loaded if you are using deepspeed.
model_name_split = script_args.model_name.split("/")[-1]
if script_args.train_obj == 'rew':
    output_name = f"{model_name_split}_peft_hh-rlhf__{script_args.train_dataset}-seed{script_args.seed}"
elif script_args.train_obj == 'dpo':
    output_name = f"{model_name_split}_peft_hh-dpo__{script_args.train_dataset}-seed{script_args.seed}"
else:
    print("WRONG TRAINING OBJECTIVE!!!")
    exit()

training_args = TrainingArguments(
    output_dir=f'{script_args.root_dir}/dahp/{"reward" if script_args.train_obj == "rew" else "aligned"}_models/{output_name}',
    learning_rate=script_args.learning_rate,
    per_device_train_batch_size=script_args.per_device_train_batch_size,
    per_device_eval_batch_size=script_args.per_device_eval_batch_size,
    num_train_epochs=script_args.num_train_epochs,
    weight_decay=script_args.weight_decay,
    eval_strategy="steps",
    eval_steps=4000, # 1000
    save_strategy="steps",
    save_steps=4000, # 1000
    gradient_accumulation_steps=script_args.gradient_accumulation_steps,
    gradient_checkpointing=script_args.gradient_checkpointing,
    deepspeed=script_args.deepspeed,
    local_rank=script_args.local_rank,
    remove_unused_columns=False,
    label_names=[],
    bf16=script_args.bf16,
    logging_strategy="steps",
    logging_steps=100,
    optim=script_args.optim,
    lr_scheduler_type=script_args.lr_scheduler_type,
    seed=script_args.seed,
)


# Load the value-head model and tokenizer.
tokenizer = AutoTokenizer.from_pretrained(f'{script_args.root_dir}/Meta-Llama-3-8B',
                                          model_max_length=script_args.max_length)
tokenizer.pad_token = tokenizer.eos_token

peft_config = LoraConfig(
    task_type=TaskType.SEQ_CLS if script_args.train_obj == 'rew' else TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=32,
    lora_dropout=0.1,
)

if script_args.train_obj == 'rew':
    model = AutoModelForSequenceClassification.from_pretrained(
        f'{script_args.root_dir}/Meta-Llama-3-8B', torch_dtype=torch.bfloat16, num_labels=1
    )
elif script_args.train_obj == 'dpo':
    model = AutoModelForCausalLM.from_pretrained(
        f'{script_args.root_dir}/Meta-Llama-3-8B', torch_dtype=torch.bfloat16
    )
else:
    print('WRONG TRAINING OBJECTIVE!!!')
    exit()

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()

# Need to do this for gpt2, because it doesn't have an official pad token.
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.eos_token_id
model.config.use_cache = not script_args.gradient_checkpointing
num_proc = 24  # Can adjust to be higher if you have more processors.
original_columns = train_dataset.column_names


# reward model definitions
# x is sequence length (number of tokens)
def r(x, i):
    assert x > 0
    assert x <= script_args.max_length
    assert 1 <= i <= script_args.num_user_types
    if i == 1:
        return min(4.0, math.exp(x / 80) / 100)
    elif i == 2:
        return min(4.0, math.exp((512 - x) / 80) / 100)
    elif i == 3:
        return math.exp(1 / (((x - 256) / 64)**2 + 1)) - 1

# Turn the dataset into pairs of query + responses, where text_c is the chosen query+response and text_r is the rejected one.
# Then tokenize the dataset.
def rnd_rew_preprocess_function(examples):
    new_examples = {
        "input_ids_c": [],
        "attention_mask_c": [],
        "input_ids_r": [],
        "attention_mask_r": [],
    }
    for response_c, response_r in zip(examples["chosen"], examples["rejected"]):
        tokenized_c = tokenizer(response_c, truncation=True, add_special_tokens=True)
        tokenized_r = tokenizer(response_r, truncation=True, add_special_tokens=True)
        if len(tokenized_c['input_ids']) > script_args.max_length or len(
                tokenized_r['input_ids']) > script_args.max_length:
            continue
        user_type = random.choice(np.arange(1, script_args.num_user_types+1))
        delta_r = r(len(tokenized_c["input_ids"]), user_type) - r(len(tokenized_r["input_ids"]), user_type)
        pref_prob = expit(delta_r)
        pref = np.random.binomial(n=1, p=pref_prob)
        if pref == 1:
            new_examples["input_ids_c"].append(tokenized_c["input_ids"])
            new_examples["attention_mask_c"].append(tokenized_c["attention_mask"])
            new_examples["input_ids_r"].append(tokenized_r["input_ids"])
            new_examples["attention_mask_r"].append(tokenized_r["attention_mask"])
        elif pref == 0:
            new_examples["input_ids_c"].append(tokenized_r["input_ids"])
            new_examples["attention_mask_c"].append(tokenized_r["attention_mask"])
            new_examples["input_ids_r"].append(tokenized_c["input_ids"])
            new_examples["attention_mask_r"].append(tokenized_c["attention_mask"])
        else:
            print("WRONG PREF")
            exit()

    return new_examples

def avg_rew_preprocess_function(examples):
    new_examples = {
        "input_ids_c": [],
        "attention_mask_c": [],
        "input_ids_r": [],
        "attention_mask_r": [],
    }
    for response_c, response_r in zip(examples["chosen"], examples["rejected"]):
        tokenized_c = tokenizer(response_c, truncation=True)
        tokenized_r = tokenizer(response_r, truncation=True)
        if len(tokenized_c['input_ids']) > script_args.max_length or len(
                tokenized_r['input_ids']) > script_args.max_length:
            continue
        delta_r_list = [r(len(tokenized_c["input_ids"]), user_type) - r(len(tokenized_r["input_ids"]), user_type) for user_type in range(1, script_args.num_user_types+1)]
        avg_delta_r = np.mean(delta_r_list)
        pref_prob = expit(avg_delta_r)
        pref = np.random.binomial(n=1, p=pref_prob)
        if pref == 1:
            new_examples["input_ids_c"].append(tokenized_c["input_ids"])
            new_examples["attention_mask_c"].append(tokenized_c["attention_mask"])
            new_examples["input_ids_r"].append(tokenized_r["input_ids"])
            new_examples["attention_mask_r"].append(tokenized_r["attention_mask"])
        elif pref == 0:
            new_examples["input_ids_c"].append(tokenized_r["input_ids"])
            new_examples["attention_mask_c"].append(tokenized_r["attention_mask"])
            new_examples["input_ids_r"].append(tokenized_c["input_ids"])
            new_examples["attention_mask_r"].append(tokenized_c["attention_mask"])
        else:
            print("WRONG PREF")
            exit()

    return new_examples

def det_avg_rew_preprocess_function(examples):
    new_examples = {
        "input_ids_c": [],
        "attention_mask_c": [],
        "input_ids_r": [],
        "attention_mask_r": [],
    }
    for response_c, response_r in zip(examples["chosen"], examples["rejected"]):
        tokenized_c = tokenizer(response_c, truncation=True)
        tokenized_r = tokenizer(response_r, truncation=True)
        if len(tokenized_c['input_ids']) > script_args.max_length or len(
                tokenized_r['input_ids']) > script_args.max_length:
            continue
        delta_r_list = [r(len(tokenized_c["input_ids"]), user_type) - r(len(tokenized_r["input_ids"]), user_type) for user_type in range(1, script_args.num_user_types+1)]
        avg_delta_r = np.mean(delta_r_list)
        pref_prob = expit(avg_delta_r)
        if pref_prob >= 0.5:
            new_examples["input_ids_c"].append(tokenized_c["input_ids"])
            new_examples["attention_mask_c"].append(tokenized_c["attention_mask"])
            new_examples["input_ids_r"].append(tokenized_r["input_ids"])
            new_examples["attention_mask_r"].append(tokenized_r["attention_mask"])
        else:
            new_examples["input_ids_c"].append(tokenized_r["input_ids"])
            new_examples["attention_mask_c"].append(tokenized_r["attention_mask"])
            new_examples["input_ids_r"].append(tokenized_c["input_ids"])
            new_examples["attention_mask_r"].append(tokenized_c["attention_mask"])

    return new_examples


def sum_rew_preprocess_function(examples):
    new_examples = {
        "input_ids_c": [],
        "attention_mask_c": [],
        "input_ids_r": [],
        "attention_mask_r": [],
    }
    for response_c, response_r in zip(examples["chosen"], examples["rejected"]):
        tokenized_c = tokenizer(response_c, truncation=True)
        tokenized_r = tokenizer(response_r, truncation=True)
        if len(tokenized_c['input_ids']) > script_args.max_length or len(
                tokenized_r['input_ids']) > script_args.max_length:
            continue
        delta_r_list = [r(len(tokenized_c["input_ids"]), user_type) - r(len(tokenized_r["input_ids"]), user_type) for user_type in range(1, script_args.num_user_types+1)]
        avg_delta_r = np.sum(delta_r_list)
        pref_prob = expit(avg_delta_r)
        pref = np.random.binomial(n=1, p=pref_prob)
        if pref == 1:
            new_examples["input_ids_c"].append(tokenized_c["input_ids"])
            new_examples["attention_mask_c"].append(tokenized_c["attention_mask"])
            new_examples["input_ids_r"].append(tokenized_r["input_ids"])
            new_examples["attention_mask_r"].append(tokenized_r["attention_mask"])
        elif pref == 0:
            new_examples["input_ids_c"].append(tokenized_r["input_ids"])
            new_examples["attention_mask_c"].append(tokenized_r["attention_mask"])
            new_examples["input_ids_r"].append(tokenized_c["input_ids"])
            new_examples["attention_mask_r"].append(tokenized_c["attention_mask"])
        else:
            print("WRONG PREF")
            exit()

    return new_examples


# preprocess the dataset and filter out QAs that are longer than script_args.max_length
if script_args.train_dataset == 'avg_rew':
    train_preprocess_function = avg_rew_preprocess_function
elif script_args.train_dataset == 'sum_rew':
    train_preprocess_function = sum_rew_preprocess_function
elif script_args.train_dataset == 'rnd_rew':
    train_preprocess_function = rnd_rew_preprocess_function
else:
    print('WRONG SCRIPT ARG: TRAIN_DATASET!!')
    exit()

train_dataset = train_dataset.map(
    train_preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)

eval_dataset = eval_dataset.map(
    det_avg_rew_preprocess_function,
    batched=True,
    num_proc=num_proc,
    remove_columns=original_columns,
)


# We need to define a special data collator that batches the data in our j vs k format.
@dataclass
class RewardDataCollatorWithPadding:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        features_c = []
        features_r = []
        for feature in features:
            features_c.append(
                {
                    "input_ids": feature["input_ids_c"],
                    "attention_mask": feature["attention_mask_c"],
                }
            )
            features_r.append(
                {
                    "input_ids": feature["input_ids_r"],
                    "attention_mask": feature["attention_mask_r"],
                }
            )
        batch_c = self.tokenizer.pad(
            features_c,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch_r = self.tokenizer.pad(
            features_r,
            padding=self.padding,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors=self.return_tensors,
        )
        batch = {
            "input_ids_c": batch_c["input_ids"],
            "attention_mask_c": batch_c["attention_mask"],
            "input_ids_r": batch_r["input_ids"],
            "attention_mask_r": batch_r["attention_mask"],
            "return_loss": True,
        }
        return batch


# Define the metric that we'll use for validation.
accuracy = evaluate.load("accuracy")


def compute_metrics(eval_pred):
    predictions, _ = eval_pred
    # Here, predictions is rewards_c and rewards_r.
    # We want to see how much of the time rewards_c > rewards_r.
    # import pdb
    # pdb.set_trace()
    predictions = np.argmax(predictions, axis=0)
    labels = np.zeros(predictions.shape)
    return accuracy.compute(predictions=predictions, references=labels)

adjust_temp = (lambda x : x * script_args.num_user_types) if script_args.train_dataset == 'sum_rew' else (lambda x : x)


class RewardTrainer(Trainer):
    # Define how to compute the reward loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        rewards_c = model(input_ids=inputs["input_ids_c"], attention_mask=inputs["attention_mask_c"])[0]
        rewards_r = model(input_ids=inputs["input_ids_r"], attention_mask=inputs["attention_mask_r"])[0]
        loss = -nn.functional.logsigmoid(adjust_temp(rewards_c - rewards_r)).mean()
        if return_outputs:
            return loss, {"rewards_c": rewards_c, "rewards_r": rewards_r}
        return loss

class DPOTrainer(Trainer):
    # Define how to compute the alignment loss.
    def compute_loss(self, model, inputs, return_outputs=False):
        logits_c = model(input_ids=inputs["input_ids_c"], attention_mask=inputs["attention_mask_c"]).logits
        log_probs_c = torch.log_softmax(logits_c, dim=-1)
        token_log_probs_c = log_probs_c.gather(2, inputs["input_ids_c"].unsqueeze(-1)).squeeze(-1)
        assert token_log_probs_c.shape == inputs['attention_mask_c'].shape
        filtered_log_probs_c = token_log_probs_c * inputs['attention_mask_c']
        logprob_c = filtered_log_probs_c.sum(dim=-1)

        logits_r = model(input_ids=inputs["input_ids_r"], attention_mask=inputs["attention_mask_r"]).logits
        log_probs_r = torch.log_softmax(logits_r, dim=-1)
        token_log_probs_r = log_probs_r.gather(2, inputs["input_ids_r"].unsqueeze(-1)).squeeze(-1)
        assert token_log_probs_r.shape == inputs['attention_mask_r'].shape
        filtered_log_probs_r = token_log_probs_r * inputs['attention_mask_r']
        logprob_r = filtered_log_probs_r.sum(dim=-1)

        loss = -nn.functional.logsigmoid(adjust_temp(logprob_c - logprob_r)).mean()
        if return_outputs:
            return loss, {"logprob_c": logprob_c, "logprob_r": logprob_r}
        return loss


# Train the model, woohoo.
if script_args.train_obj == 'rew':
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
    )
elif script_args.train_obj == 'dpo':
    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        data_collator=RewardDataCollatorWithPadding(tokenizer=tokenizer),
    )
else:
    print('WRONG TRAINING OBJECIVE!!!')
    exit()

if script_args.eval_first_step:

    class EvaluateFirstStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == 1:
                control.should_evaluate = True

    trainer.add_callback(EvaluateFirstStepCallback())

trainer.train(script_args.resume_from_checkpoint)

print("Saving last checkpoint of the model")
model.save_pretrained(output_name + "_peft_last_checkpoint")
