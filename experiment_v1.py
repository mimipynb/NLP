""" 

    experiment_v1.py dated 16.01.2025
    
    Description:
        Fine-tuning instruct LLM already trained with many languages. Given that the text generation noise is already incooperated within LLM hidden layers then I would want to use this as a base model for comparing new, unseen queries in the classification stage. 

    TODO:
    [x] Data Analysis 
    [x] Model embedding multilingual
    [x] Feature / Model selection 
    [x] Data class iterator and loader setup 
    [ ] Model classification 
"""


import wandb 
import torch 
import pandas as pd 
from pathlib import Path 
from datasets import Dataset 
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

device = torch.device(
    'mps' if torch.backends.mps.is_available() 
    else 'cuda' if torch.cuda.is_available() 
    else 'cpu'
)

try:
    # Loads the dataset from path
    data_path = {}
    for folder in Path('wsdm-cup-multilingual-chatbot-arena').iterdir():
        data_path[folder.stem] = folder.resolve()
        
    if not data_path:
        raise OSError(f"Loading File Error data_path is empty. Data: {data_path}")
    else:
        OUTPUT_PATH = Path("output").resolve()
        print('All data loaded: ', data_path, '\nOutput path: ', OUTPUT_PATH)
        ds = pd.read_parquet(data_path['train'])
        ds = ds.sample(n=100, random_state=42)
        submission_ds = pd.read_parquet(data_path['test'])
        
except Exception as e:
    print(e)
    raise e 

# Start a W&B Run with wandb.init
run = wandb.init(
    project="wsdm-cup-multilingual-chatbot-arena_experiment",
    name="experiment_17-01-2025"
)

model_card = "HuggingFaceTB/SmolLM-135M-Instruct"
adapter_name = "adapter_1"
layers = ['q_proj', 'k_proj', 'v_proj']

model_kwargs = dict(
    use_cache=False,
    trust_remote_code=True,
    torch_dtype=torch.float32,
    device_map=None
)

peft_config = LoraConfig(
    r=8, 
    lora_alpha=16, 
    lora_dropout=0.10, 
    bias="none", 
    inference_mode=False,
    task_type=TaskType.CAUSAL_LM,
    target_modules=layers,
    init_lora_weights='gaussian'
)

generation_config = dict(
    do_sample=True, 
    # sampling params
    temperature=0.8, 
    top_k=60, 
    top_p=0.8, 
    repetition_penalty=1.2,
    # tokens generation conditions
    max_new_tokens=50, 
    num_return_sequences=1,
    use_cache=False
)

run.config.peft_config = peft_config 
run.config.generation_config = generation_config 
run.config.model_kwargs = model_kwargs 

tokenizer = AutoTokenizer.from_pretrained(model_card)
model = AutoModelForCausalLM.from_pretrained(model_card, **model_kwargs).to(device)
peft_model = get_peft_model(model, peft_config, adapter_name).to(device)

def dataPreprocess(data):
    """ Prepares / preprocess the training datasets. """
    
    for _, item in data.iterrows():
        response = item['response_a'] if item['winner'] == 'model_a' else item['response_b']
        
        inputs = tokenizer.apply_chat_template(
            [
                {"role": "user", "content": item['prompt']},
                {"role": "assistant", "content": response}
            ],
            tokenize=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            return_dict=True,
            return_attention_mask=True, 
            add_generation_prompt=False,
        )
        
        if not all(key in inputs for key in ["input_ids", "attention_mask"]):
            raise ValueError(f"Missing keys in tokenizer output for item: {item}")
        
        inputs['input_ids'] = inputs['input_ids'].squeeze(0)
        inputs['attention_mask'] = inputs['attention_mask'].squeeze(0)
        
        yield inputs

# Preprocesses train and eval dataset
data = Dataset.from_generator(lambda: dataPreprocess(ds))
data.set_format(type='torch', columns=["input_ids", "attention_mask"])
data_ = data.train_test_split(test_size=0.65, shuffle=True, seed=42)
train_data, eval_data = data_["train"], data_["test"]

def create_training_arguments(path, learning_rate=0.0035, epochs=3):
    training_args = TrainingArguments(
        # Directory args
        output_dir=path,  # Output path
        overwrite_output_dir=True,  # Always overwrite
        load_best_model_at_end=True,  # Only works if eval strategy is active
        
        # Training behavior
        gradient_accumulation_steps=1,
        auto_find_batch_size=True,  # Auto-adjust batch size
        
        # Training args
        warmup_ratio=0.1,
        learning_rate=learning_rate,
        weight_decay=0.0,
        num_train_epochs=epochs,
        max_grad_norm=1.0,
        
        # Evaluation and Saving strategies (MUST MATCH)
        report_to="wandb",
        save_strategy="steps",  # Save the model at intervals
        eval_strategy="steps",
        # Optimization args
        optim="adamw_torch"
    )
    return training_args

def create_trainer(model, training_args, train_dataset, eval_dataset):
    trainer = Trainer(
        model=model,  # ensure to pass peft model
        args=training_args, 
        train_dataset=train_dataset, 
        eval_dataset=eval_dataset
    )
    return trainer

train_config = create_training_arguments(OUTPUT_PATH, learning_rate=0.001, epochs=12)
trainer = create_trainer(peft_model, train_config, train_data, eval_data)
run.config.train_config = train_config

trainer.train()

run.finish()
