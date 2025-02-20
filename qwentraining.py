from unsloth import FastLanguageModel
import torch
from datasets import load_dataset, concatenate_datasets

max_seq_length = 2048 # Choose any! We auto support RoPE Scaling internally!
dtype = None # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
DATASET_NAMES = ["Pradeep016/career-guidance-qa-dataset", "pranayvadla17/learningpath"]
OUTPUT_DIR = "./career-qwen"

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = "unsloth/Qwen2.5-14B-Instruct-1M-bnb-4bit",
    max_seq_length = max_seq_length,
    dtype = dtype,
    load_in_4bit = load_in_4bit,
)

model = FastLanguageModel.get_peft_model(
    model,
    r = 16, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj",],
    lora_alpha = 16,
    lora_dropout = 0, # Supports any, but = 0 is optimized
    bias = "none",    # Supports any, but = "none" is optimized
    # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA
    loftq_config = None, # And LoftQ
)

alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

EOS_TOKEN = tokenizer.eos_token # Must add EOS_TOKEN

# 1. Load and prepare datasets (Career QA)
def format_qa_dataset(examples):
    instructions = ["Answer this career question about " + role for role in examples["role"]]
    inputs       = examples["question"]
    outputs      = examples["answer"]
    
    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)

    return { "text": texts }


# 2. Load and prepare datasets (Learning Path)
def format_lp_dataset(examples):
    instructions = ["You are a career and learning path recommender that needs to recommend learning paths for users"] * len(examples["Level"])
    inputs       = ["Provide the " + level + "-level learning path for " + course 
                    for level, course in zip(examples["Level"], examples["CourseName"])]
    outputs      = examples["Learning Path"]

    texts = []
    for instruction, input_text, output in zip(instructions, inputs, outputs):
        text = alpaca_prompt.format(instruction, input_text, output) + EOS_TOKEN
        texts.append(text)

    return { "text": texts }


# Load datasets
qa_dataset = load_dataset(DATASET_NAMES[0], split="train")
lp_dataset = load_dataset(DATASET_NAMES[1])

# Combine train and test of lp_dataset
lp_combined = concatenate_datasets([lp_dataset["train"], lp_dataset["test"]])

# Format datasets
qa_dataset = qa_dataset.map(format_qa_dataset,batched=True)
lp_combined = lp_combined.map(format_lp_dataset, batched=True)

# Combine datasets
combined_dataset = concatenate_datasets([qa_dataset, lp_combined]).shuffle(seed=42)

# # Split dataset into train and test
# combined_dataset = combined_dataset.train_test_split(test_size=0.1)
# train_dataset = combined_dataset["train"]
# eval_dataset = combined_dataset["test"]

from trl import SFTTrainer
from transformers import TrainingArguments
from unsloth import is_bfloat16_supported

trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = combined_dataset,
    dataset_text_field = "text",
    max_seq_length = max_seq_length,
    dataset_num_proc =1,
    packing = False, # Can make training 5x faster for short sequences.
    args = TrainingArguments(
        per_device_train_batch_size = 2,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs = 1, # Set this for 1 full training run.
 
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 1,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none", # Use this for WandB etc
    ),
)

trainer_stats = trainer.train()

model.save_pretrained(OUTPUT_DIR) 
tokenizer.save_pretrained(OUTPUT_DIR)