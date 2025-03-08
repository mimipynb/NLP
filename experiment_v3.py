""" 

    experiment_v2.py dated 18.01.2025 
    
    Description
        Training Script for utilizing Sentence Transformer library. 
    
    Sources:
        - Sentence Encoder Card: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        - Prompts training example: https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/prompts/training_nq_prompts.py
"""

import random 
from pathlib import Path 
from datetime import datetime

import wandb
import torch 
import numpy as np  
import pandas as pd 
from datasets import Dataset

from sentence_transformers.evaluation import NanoBEIREvaluator, EmbeddingSimilarityEvaluator
from sentence_transformers.training_args import BatchSamplers
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
    SentenceTransformerModelCardData,
    losses, 
)

import os
import warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=UserWarning, module="pydantic")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
device = torch.device(
    'mps' if torch.backends.mps.is_available() 
    else 'cuda' if torch.cuda.is_available() 
    else 'cpu'
)

comp_data_name = "wsdm-cup-multilingual-chatbot-arena"
model_card = "paraphrase-multilingual-MiniLM-L12-v2"
run_name = comp_data_name + datetime.now().strftime("%m-%d-%H:%M:%S") 

use_prompts = True
include_prompts_in_pooling = True
data_cols = ['prompt', 'language', 'sentence1', 'sentence2', 'label']
query_prompt = "prompt: "
# query_language = "language: "
query_response = "response: "
prompts_arg = {
	"prompt": query_prompt,
	# "language": query_language,
    "sentence1": query_response,
    "sentence2": query_response
}

wandb.init(project=comp_data_name, name=run_name)

try:
    # 1. Loads the dataset from path
    data_path = {}
    for folder in Path(comp_data_name).iterdir():
        data_path[folder.stem] = folder.resolve()
        
    if not data_path:
        raise OSError(f"Loading File Error data_path is empty. Data: {data_path}")
    else:
        OUTPUT_PATH = Path("output").resolve()
        wandb.log({"info": "All data loaded: {data_path} and Output path: {OUTPUT_PATH}"})
        ds = pd.read_parquet(data_path['train'])
        ds = ds.sample(n=100, random_state=SEED) # TODO: To be removed when actually learning the sequence or submitting 
        ds.reset_index(drop=True, inplace=True)
        submission_ds = pd.read_parquet(data_path['test'])
        
except Exception as e:
    wandb.log({"error": e})
    raise

# cols: ['id', 'prompt', 'response_a', 'response_b', 'winner', 'model_a', 'model_b', 'language']

# 2. Loads the model and training params

model_data_card = SentenceTransformerModelCardData(
    model_name=model_card,
    license='mit',
    task_name="semantic textual similarity, sem, classification",
)

model = SentenceTransformer(model_card, model_data_card, device=device)
#model.set_pooling_include_prompt(include_prompts_in_pooling)
wandb.config.update({
    'model_card': model_card, 
    'prompts': prompts_arg,
    'model_data_card': model_data_card,
})

# 3. Preprocesses the dataset
def prepareDataset(row):
    row['sentence1'] = row['response_a'] if row['winner'] == 'model_a' else row['response_b']
    row['sentence2'] = row['response_a'] if row['winner'] == 'model_b' else row['response_b']

    #win_response = model.encode(row['win_response'])
    #lose_response = model.encode(row['lose_response'])
    #score = model.similarity(win_response, lose_response)
    #assert len(score) == 1 and len(score[0]) == 1, logging.error(f'Error score returned: {score}')
    #row['score'] = -score.squeeze(0).squeeze(0)
    row['label'] = 0
    
    return row 

data = Dataset.from_pandas(ds[['prompt', 'language', 'response_a', 'response_b', 'winner']])
data = data.map(prepareDataset, batched=False, remove_columns=['response_a', 'response_b', 'winner', 'language', 'prompt'])
wandb.log({"info": f"Completed preprocessing dataset. Data columns: {data[0].keys()}"})
print(data[0].keys())
data = data.train_test_split(test_size=0.3)
train_data, eval_data = data['train'], data['test']

# Defining the criterion and batch size
# loss = CachedMultiplesentence2sRankingLoss(model, mini_batch_size=16)
# loss = losses.Multiplesentence2sRankingLoss(model=model)
wandb.log({"info": "Starting training now ..."})

loss = losses.ContrastiveLoss(model, distance_metric=losses.SiameseDistanceMetric.COSINE_DISTANCE, margin=0.7)

train_args = SentenceTransformerTrainingArguments(
    output_dir=OUTPUT_PATH / run_name, 
    overwrite_output_dir=True,
    # Optional training parameters:
    num_train_epochs=1,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    learning_rate=1e-6,
    warmup_ratio=0.01,
    # Optional tracking/debugging parameters:
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    save_total_limit=2,
    logging_steps=5,
    logging_first_step=True,
    run_name=run_name,  # Will be used in W&B if `wandb` is installed
    seed=SEED,
    # Prompts
    prompts=prompts_arg, 
)
wandb.config.train_config = train_args 
dev_evaluator = EmbeddingSimilarityEvaluator(
    batch_size=train_args.per_device_eval_batch_size,
    sentences1=eval_data["sentence1"],
    sentences2=eval_data["sentence2"],
    scores=eval_data["label"],
    name="sts_dev",
    show_progress_bar=True, 
)

# 7. Create a trainer & train
trainer = SentenceTransformerTrainer(
    model=model,
    args=train_args,
    train_dataset=train_data,
    eval_dataset=eval_data,
    loss=loss,
    evaluator=dev_evaluator,
)

trainer.train()
results = trainer.evaluate(eval_data)
wandb.log({"info": f"Evaluation results: {results}"})
print(results)
# test_results = trainer.predict(Dataset.from_pandas(submission_ds[['response_a', 'response_b']]))
# wandb.log({"info": f"Prediction results: \n{test_results}"})
# trainer.predict(submission_ds, label_ids=['prompt', 'response_a', 'response_b'])
wandb.finish()

