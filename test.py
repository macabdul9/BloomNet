import torch
from transformers import AutoTokenizer
from config import config
from dataset.loader import get_loaders
from config import config
from evaluation import evaluate
import pandas as pd
import torch.nn as nn
from models.baselines.LSTM import LSTMClassifier
from models.baselines.LSTMAttn import LSTMAttnClassifier
from models.baselines.CNN import CNNClassifier
from models.baselines.VDCNN import VDCNNClassifier
from models.baselines.RCNN import RCNNClassifier
from models.baselines.SelfAttn import SelfAttnClassifier


 
import torch.nn.functional as F

if __name__ == '__main__':
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    loaders = get_loaders(tokenizer=tokenizer, config=config['data'])
    
    # model = VDCNNClassifier(
    #     vocab_size=tokenizer.vocab_size,
    #     # num_labels=6
    # )
    
    model = SelfAttnClassifier(
        vocab_size=tokenizer.vocab_size, 
        hidden_size=768,
        num_classes=6
    )

    batch = next(iter(loaders['fold0']['train']))
    
    logits = model.forward(
        input_ids=batch['input_ids'],
        # _len = batch['_len']
    )     # packed_output, (ht, ct)

    # print(f'packed_output.shape = {packed_output.shape}, ht.shape = {ht.shape}, ct.shape = {ct.shape}')
    
    
    # ground_truth, predicted_class, f1, accuracy, cr = evaluate(model=model, loader=loaders['fold0']['train'], device=device)
    
    
    # df = pd.DataFrame(
    #     data={https://www.overleaf.com/project/60ab65e1d0322cadbd7bf947
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
    
    