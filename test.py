import torch
from transformers import AutoTokenizer
from config import config
from dataset.loader import get_loaders
from models.baselines.LSTM import LSTMClassifier
from config import config
from evaluation import evaluate
import pandas as pd
import torch.nn as nn
 
import torch.nn.functional as F

if __name__ == '__main__':
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    loaders = get_loaders(tokenizer=tokenizer, config=config['data'])
    
    model = LSTMClassifier(
        batch_size=config['data']['batch_size'],
        output_size=config['data']['num_classes'],
        hidden_size=config['model']['hidden_size'],
        vocab_size=tokenizer.vocab_size,
        embedding_length=config['model']['hidden_size'],
    )
    

    batch = next(iter(loaders['fold0']['train']))
    
    logits = model.forward(
        input_ids=batch['input_ids']
    )
    
    print(batch['target'])
    
    
    # ground_truth, predicted_class, f1, accuracy, cr = evaluate(model=model, loader=loaders['fold0']['train'], device=device)
    
    
    # df = pd.DataFrame(
    #     data={
    #         "Ground_Truth":ground_truth,
    #         "Predicted_Class":predicted_class
    #     }
    # )
    
    # df.to_csv("results.csv", index=False)
    
    # print(cr)
    
    # loss = nn.CrossEntropyLoss()
    
    # l = loss(logits, batch['target'].squeeze())
    
    # print(l.data)
    # print(f1, accuracy)
    
    print(f'logits.shape = {logits.shape}')
    
    