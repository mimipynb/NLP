""" 

    experiment_v2.py dated 18.01.2025 
    
    Description
        With custom models built with pytorch and tensors handling.
    
    Sources:
        - Sentence Encoder Card: https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
        
"""

import pandas as pd 
from pathlib import Path 

# import wandb 
import torch 
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader, random_split
from sentence_transformers import SentenceTransformer, SimilarityFunction

device = torch.device(
    'mps' if torch.backends.mps.is_available() 
    else 'cuda' if torch.cuda.is_available() 
    else 'cpu'
)

kaggle_data_name = 'wsdm-cup-multilingual-chatbot-arena'
sample_subset_size = 2000 
seed = 42

try:
    # Loads the dataset from path
    data_path = {}
    for folder in Path(kaggle_data_name).iterdir():
        data_path[folder.stem] = folder.resolve()
        
    if not data_path:
        raise OSError(f"Loading File Error data_path is empty. Data: {data_path}")
    else:
        OUTPUT_PATH = Path("output").resolve()
        OUTPUT_PATH.mkdir(exist_ok=True)
        print('All data loaded: ', data_path, '\nOutput path: ', OUTPUT_PATH)
        
        # loads training dataset
        ds = pd.read_parquet(data_path['train'])
        ds = ds.sample(n=sample_subset_size, random_state=seed)
        ds.reset_index(drop=True, inplace=True)
        
        # loads submission test dataset
        submission_ds = pd.read_parquet(data_path['test'])
        
except Exception as e:
    print(e)

class MiData(Dataset):
    def __init__(self, prompt, response, target):
        assert prompt.shape == response.shape, "Expected matching shapes for input prompts and response"
        assert target.shape[0] == response.size(0) == prompt.size(0), "Expected target values available for each data row"
        
        self.prompt = prompt 
        self.response = response
        self.target = target 
        
    def __len__(self):
        return len(self.prompt)
    
    def __getitem__(self, index):
        prompt = self.prompt[index]
        text = self.response[index]
        target = self.target[index].float()

        return prompt, text, target 

def create_target_vectors(data):
    target_A = []
    target_B = []
    
    for _, row in data.iterrows():
        if row['winner'] == 'model_a':
            target_A += [1]
            target_B += [0]
        else:
            target_A += [0]
            target_B += [1]
            
    target_A = torch.tensor(target_A, requires_grad=False).unsqueeze(0)
    target_B = torch.tensor(target_B, requires_grad=False).unsqueeze(0)
    
    return target_A, target_B

def prepareDataset(data: pd.DataFrame, model: SentenceTransformer):
    data['winner_index'] = [0 if i['winner'] == 'model_a' else 1 for _, i in data.iterrows()] # 1 = model_b and 0 = model_a
    data['winning_model'] = [i['model_a'] if i['winner'] == 'model_a' else i['model_b'] for _, i in data.iterrows()]
    
    target_a, target_b = create_target_vectors(data)
    prompt_embedding = model.encode(data['prompt'].values, convert_to_tensor=True, device=device)
    response_a_embedding = model.encode(data['response_a'].values, convert_to_tensor=True, device=device)
    response_b_embedding = model.encode(data['response_b'].values, convert_to_tensor=True, device=device)
    
    X_prompt = torch.vstack([prompt_embedding, prompt_embedding])
    X_response = torch.vstack([response_a_embedding, response_b_embedding])
    y = torch.concat([target_a, target_b]).flatten()

    return MiData(X_prompt, X_response, y)

def train_test_split(dataset, train_size: float = 0.7):
    train_size_data = int(train_size * len(dataset))
    test_size = len(dataset) - train_size_data 
    return random_split(dataset, [train_size_data, test_size])

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.rnn_encoder_config = dict(
            input_size=384, 
            hidden_size=384 // 3, 
            num_layers=2, 
            nonlinearity='tanh', # default 
            bias=True, 
            dropout=0.0001, 
            batch_first=True
        )
        
        self.class_config = dict(
            in_features=384 // 3, 
            out_features=1, 
            bias=True
        )
        
        self.prompt_function = nn.RNN(**self.rnn_encoder_config)
        self.input_function = nn.RNN(**self.rnn_encoder_config)
        self.function = nn.Linear(**self.class_config)
        
        self.prompt_hidden = None 
        self.input_hidden = None 
        self.to(device)
        
    def forward(self, prompt, inputs):
        # Move inputs to device
        prompt = prompt.to(device)
        inputs = inputs.to(device)
        
        if not self.training:
            z_prompt, _ = self.prompt_function(prompt)  # Forward pass without hidden states
            x_input, _ = self.input_function(inputs)   # Use `inputs` instead of `input`
        else:
            if self.prompt_hidden is None or self.input_hidden is None:
                size = (self.rnn_encoder_config['num_layers'], self.rnn_encoder_config['hidden_size'])
                self.prompt_hidden = torch.zeros(size, device=device)
                self.input_hidden = torch.zeros(size, device=device)

                # Detach hidden states to avoid retaining computation history
            self.prompt_hidden = self.prompt_hidden.detach()
            self.input_hidden = self.input_hidden.detach()

            # Forward pass with hidden states
            z_prompt, self.prompt_hidden = self.prompt_function(prompt, self.prompt_hidden)
            x_input, self.input_hidden = self.input_function(inputs, self.input_hidden)  # Use `inputs` here

        print(f"Encoded Prompt shape: {z_prompt.shape} ({self.prompt_hidden.size()}) & Input shape: {x_input.size()} (Hidden {self.input_hidden.size()})", end="\n \t")
        
        x_stack = torch.concat([z_prompt.unsqueeze(0).permute(1, 0, 2), x_input.unsqueeze(0).permute(1, 0, 2)], dim=1).mean(dim=1).tanh() # (batch_size, hidden_size * 2)
        print(f"Concatenated shape: {x_stack.shape}", end="\n \t")

        output = self.function(x_stack).to(device)
        print(f"Output (shape: {output.shape})", end="\n \t")
        return output.squeeze(1)

def trainer(loader, model, num_epochs=2):
    optim_config = dict(
        lr=0.001, 
        eps=1e-08, 
        weight_decay=0,  # weight decay - l2 penalty default 
        maximize=False, # instead of minimizing the objective wrt params 
        differentiable=False # defaults to false where torch.no_grad() will be used when stepping every optim 
        
    )
    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.Adam(model.parameters(), **optim_config)
    
    for epoch in range(num_epochs):
        print(f"Epoch [{epoch+1}/{num_epochs}]", end=" ", flush=True)
        
        for batch_num, (prompts, inputs, targets) in enumerate(loader):
            print(f"Batch input size 32 n.o {batch_num+1}", end=" ", flush=True)
            prompts, inputs, targets = prompts.to(device), inputs.to(device), targets.to(device)
            output = model(prompts, inputs)
            loss = criterion(output, targets)
            print(f"Loss: {loss:.5f}", end="\n")
            
            loss.backward(retain_graph=True)
            optim.step()
            optim.zero_grad()

def evaluate(model, loader, criterion):
    """
    Evaluate the model on a given dataset.

    Args:
        model (nn.Module): The trained PyTorch model.
        loader (DataLoader): DataLoader for the test dataset.
        criterion (nn.Module): Loss function used for evaluation.

    Returns:
        float: Average loss over the dataset.
        float: Accuracy of the model on the dataset.
    """
    model.eval()  # Set model to evaluation mode
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():  # Disable gradient calculation for evaluation
        for batch_num, (prompts, inputs, targets) in enumerate(loader):
            prompts, inputs, targets = prompts.to(device), inputs.to(device), targets.to(device)

            # Forward pass
            output = model(prompts, inputs)
            # Compute loss
            loss = criterion(output, targets)
            total_loss += loss.item()

            # Compute predictions and accuracy
            predictions = (output.flatten() > 0.5).long()  # Binary classification (threshold = 0.5)
            correct += (predictions == targets).sum().item()
            total += targets.size(0)

            print(f"Batch {batch_num+1}: Loss = {loss.item():.5f}")

    # Calculate average loss and accuracy
    avg_loss = total_loss / len(loader)
    accuracy = correct / total

    print(f"\nEvaluation Results: Average Loss = {avg_loss:.5f}, Accuracy = {accuracy:.5%}")
    return avg_loss, accuracy

if __name__ == '__main__':
    # Setting the model card / configs for Sentence Transformer
    model_card = "paraphrase-multilingual-MiniLM-L12-v2"
    encoder = SentenceTransformer(model_card, similarity_fn_name=SimilarityFunction.COSINE)
    data = prepareDataset(ds, encoder)
    
    train_data, test_data = train_test_split(dataset=data, train_size=0.7)
    train_loader = DataLoader(dataset=train_data, shuffle=True, batch_size=32)
    test_loader = DataLoader(dataset=test_data)
    
    clf = Classifier()
    trainer(train_loader, clf)
    evaluate(clf, test_loader, nn.CrossEntropyLoss())