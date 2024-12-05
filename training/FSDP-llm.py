


import torch
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from datasets import load_dataset
from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
from trl import SFTTrainer

# from huggingface_hub import login
# login(
#   token="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", # ADD YOUR TOKEN HERE
# )


# Initialize FSDP with configurations for state dictionary and optimizer state handling
fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
)

# Setup the Accelerator with the FSDP plugin
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

# Base model configuration for 4-bit quantization
base_model_id = "meta-llama/Meta-Llama-3-8B"
# base_model_id = "meta-llama/Meta-Llama-3-8B-Instruct"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
)

# Load the pretrained model with quantization and device mapping
model = AutoModelForCausalLM.from_pretrained(base_model_id,
                                             quantization_config=bnb_config,
                                             device_map="auto",
                                             cache_dir='')

# Initialize tokenizer with special tokens
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id,
    add_eos_token=True,
    add_bos_token=True, 
)

# Load dataset
# dataset_name = "hf_dataset"

# dataset_name = "Hemanth-thunder/tamil-open-instruct-v1"
# dataset_name = 'AlexanderDoria/novel17_test'
# dataset_name = "VMware/open-instruct"
dataset_name = "ruslanmv/ai-medical-dataset"
dataset = load_dataset(dataset_name, split="train")

# Enable gradient checkpointing to save memory
model.gradient_checkpointing_enable()

# Prepare model for training with K-bit quantization techniques
model = prepare_model_for_kbit_training(model)


# Configuration for LoRA augmentation
config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "w1",
        "w2",
        "w3",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

# Apply LoRA and other modifications for PEFT
model = get_peft_model(model, config)

# Prepare model for distributed training with Accelerator
model = accelerator.prepare_model(model)

# Enable model parallelism if multiple GPUs are available
if torch.cuda.device_count() > 1:
    model.is_parallelizable = True
    model.model_parallel = True


# Ensure padding token is set for the tokenizer
tokenizer.pad_token = tokenizer.eos_token

# tokenized_datasets = tokenized_datasets.remove_columns_(books_dataset["train"].column_names)


# Set up the Trainer with training arguments
# trainer = SFTTrainer(
#     model=model,
#     train_dataset=dataset,
#     peft_config=config,
#     dataset_text_field="context",
#     max_seq_length=512,
#     tokenizer=tokenizer,
#     packing=True,
trainer = transformers.Trainer(
    model=model,
    train_dataset=dataset,
    args=transformers.TrainingArguments(
        output_dir='./results',
        warmup_steps=2,
        per_device_train_batch_size=16,
        gradient_checkpointing=True,
        gradient_accumulation_steps=8,
        num_train_epochs=3,
        learning_rate=2.5e-5,
        logging_steps=1,
        fp16=True, 
        optim="paged_adamw_8bit",
        logging_dir="./logs",
        save_strategy="steps",
        save_steps=100,
        save_total_limit=2,
        remove_unused_columns=False,
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)


# Disable caching to manage memory better during training
model.config.use_cache = False

# Start training
trainer.train()