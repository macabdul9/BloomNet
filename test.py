import torch
from transformers import AutoTokenizer
from config import config
from dataset.loader import get_loaders
from models.baselines.LSTM_old import LSTMClassifier
from config import config
from evaluation import evaluate
import pandas as pd
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
 
import torch.nn.functional as F

if __name__ == '__main__':
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    loaders = get_loaders(tokenizer=tokenizer, config=config['data'])
    
    model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        num_layers=4
    )
    

    batch = next(iter(loaders['fold0']['train']))
    
    logits = model.forward(
        input_ids=batch['input_ids'],
        _len = batch['_len']
    )
    
    # packed_output, output_lengths = pad_packed_sequence(packed_output, batch_first=True)

    # print(f'packed_output.shape = {packed_output.shape}, ht.shape = {ht.shape}, ct.shape = {ct.shape}')
    
    
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
    
    