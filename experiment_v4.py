
import numpy as np 
import pandas as pd 
from datasets import Dataset

from sklearn.cluster import AgglomerativeClustering
from sentence_transformers import SentenceTransformer, SentenceTransformerTrainer, losses
from sentence_transformers.evaluation import BinaryClassificationEvaluator, EmbeddingSimilarityEvaluator

model_card = "paraphrase-multilingual-MiniLM-L12-v2"
model = SentenceTransformer(model_card)

data_path = "/Users/mimiphan/Projects/wsdm-cup-multilingual-chatbot-arena/train.parquet"
ds = pd.read_parquet(data_path).sample(2000) 
ds.reset_index(drop=True, inplace=True)

# sentence gives the winning text responses only 
ds['sentence'] = [i['prompt'] + i['response_a'] if i['winner'] == 'model_a' else i['prompt'] + i['response_b'] for _, i in ds.iterrows()] # winning data rows
ds['sentence2'] = [i['prompt'] + i['response_b'] if i['winner'] == 'model_a' else i['prompt'] + i['response_a'] for _, i in ds.iterrows()] # losing data rows

# labels are all ones since we are trying to rank this higher 
ds['label'] = np.ones(ds.shape[0], dtype=int)

"""
prompt_embedding = model.encode(ds['prompt'])
ds['label'] = AgglomerativeClustering(
    n_clusters=None,
    linkage='single',
    distance_threshold=0.25, 
    compute_distances=False,
    metric="cosine", 
).fit_predict(prompt_embedding)
""" 
winning_embedding = model.encode(ds['sentence'])
losing_embedding = model.encode(ds['sentence2'])
ds['scores'] = model.similarity_pairwise(winning_embedding, losing_embedding)

data = Dataset.from_pandas(ds[['sentence', 'label']])
data = data.train_test_split(train_size=0.7)
train_data, eval_data = data['train'], data['test']

loss = losses.BatchHardTripletLoss(model)
trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_data,
    eval_dataset=eval_data,
    loss=loss
)
trainer.train()
trainer.evaluate()

# need to use Dataset with these libraries lmao ffs kms 
test_data = Dataset.from_pandas(ds)
binary_acc_evaluator = BinaryClassificationEvaluator(
    sentences1=test_data["response_a"],
    sentences2=test_data["response_b"],
    labels=test_data["label"],
    name="evaluation",
    show_progress_bar=True
)
embedding_eval = EmbeddingSimilarityEvaluator(
    sentences1=test_data['response_a'], 
    sentences2=test_data['response_b'], 
    scores=test_data['scores'], 
    name='embedding_evaluation',
    show_progress_bar=True
)

# E.g. 0: sports, 1: economy, 2: politics
train_dataset = Dataset.from_dict({
    "sentence": [
        "He played a great game.",
        "The stock is up 20%",
        "They won 2-1.",
        "The last goal was amazing.",
        "They all voted against the bill.",
    ],
    "label": [0, 1, 0, 0, 2],
})

loss = losses.BatchHardTripletLoss(model)

trainer = SentenceTransformerTrainer(
    model=model,
    train_dataset=train_dataset,
    loss=loss,
)
trainer.train()

