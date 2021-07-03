import torch
from transformers import AutoTokenizer
from config import config
from dataset.loader import get_loaders
from config import config
from evaluation import evaluate
import pandas as pd
import geoopt
import torch.nn as nn
from models.baselines.LSTM import LSTMClassifier
from models.baselines.LSTMAttn import LSTMAttnClassifier
from models.baselines.CNN import CNNClassifier
from models.baselines.VDCNN import VDCNNClassifier
from models.baselines.RCNN import RCNNClassifier
from models.baselines.SelfAttn import SelfAttnClassifier
from models.baselines.Seq2SeqAttn import Seq2SeqAttnClassifier
from models.baselines.HAN import HANClassifier
import torch.nn.functional as F


# hyperbolic model or our model
# from models.model import Model
 
import torch.nn.functional as F
from config import config

if __name__ == '__main__':
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    
    loaders = get_loaders(tokenizer=tokenizer, config=config['data'])
    
    # model = VDCNNClassifier(
    #     vocab_size=tokenizer.vocab_size,
    #     # num_labels=6
    # )



    
    # model = HANClassifier(
    #     ntoken=tokenizer.vocab_size,
    #     num_class=6
    # )

    batch = next(iter(loaders['fold0']['train']))


    # embedding = Embedding(
    #     num_embeddings=tokenizer.vocab_size, 
    #     embedding_dim=768, 
    #     manifold=geoopt.PoincareBall(c=1.0),

    # )
    
    # hygru = MobiusGRU(
    #     input_size=768,
    #     hidden_size=768,
    #     num_layers=1,
    #     bias=True,
    #     nonlin=None,
    #     hyperbolic_input=True,
    #     hyperbolic_hidden_state0=True,
    #     num_classes=6
    # )

    # input = embedding(
    #     input=batch['input_ids']
    # )

    # logits, out, ht = hygru(input)

    # model = Model(
    #     vocab_size=tokenizer.vocab_size,
    #     num_layers=1,
    #     bias=True,
    #     nonlin=None,
    #     hyperbolic_input=True,
    #     hyperbolic_hidden_state0=True,
    #     num_classes=6,
    #     input_size=768,
    #     hidden_size=768,
    #     c=1.0,
    # )



    # logits = model(
    #     input_ids=batch['input_ids'],
    #     attention_mask=batch['attention_mask'],
    #     _len=batch['_len']
    # )


    # model = HANClassifier(
    #     num_classes=config['data']['num_classes'],
    #     vocab_size=tokenizer.vocab_size,
    #     embed_dim=config['model']['hidden_size'],
    #     word_gru_hidden_dim=config['model']['hidden_size'],
    #     sent_gru_hidden_dim=config['model']['hidden_size'],
    #     word_gru_num_layers=config['model']['num_layers'],
    #     sent_gru_num_layers=config['model']['num_layers'],
    #     word_att_dim=config['model']['hidden_size'],
    #     sent_att_dim=config['model']['hidden_size'],
    #     use_layer_norm=True,
    #     dropout=config['model']['dropout']

    # )

    # # print(model)

    # docs = batch['input_ids'].unsqueeze(1)
    # doc_lengths = torch.ones(16, dtype=torch.long)
    # sent_lengths = batch['_len'].unsqueeze(1)


    # scores, word_att_weights, sent_att_weights = model(
    #     input_ids=batch['input_ids'],
    #     attention_mask=batch['attention_mask'],
    #     _len=batch['_len']
    # )

    # print(scores.shape)
    # # print(docs.shape, doc_lengths.shape, sent_lengths.shape)


    for fold in loaders:
        print("test-1")
        for i, batch in enumerate(loaders[fold]['test1']):
            print(f'batch = {i} | input.shape = {batch["input_ids"].shape} | target.shape = {batch["target"].shape}')

        print("test-2")
        for i, batch in enumerate(loaders[fold]['test2']):
            print(f'batch = {i} | input.shape = {batch["input_ids"].shape} | target.shape = {batch["target"].shape}')
        break



    # print(F.cross_entropy(logits, batch['target'].squeeze()))


    # print(type(logits))

    # print(f'out.shape = {out.shape}, ht.shape = {ht.shape}, logits.shape = {logits.shape}')

    # logits = model.forward(
    #     batch_reviews=batch['input_ids'], 
    #     sent_order=torch.linspace(0, batch['input_ids'].shape[0], batch['input_ids'].shape[0], dtype=torch.long), 
    #     ls=batch['_len'],
    #     lr=batch['_len']
    # )    

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
    
    # print(f'logits.shape = {logits.shape}')
    
    