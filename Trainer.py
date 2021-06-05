# utils
import torch

# data


# models
import torch.nn as nn

# training and evaluation
import wandb
import pytorch_lightning as pl
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score
from models.baselines.LSTM import LSTMClassifier

class LightningModel(pl.LightningModule):

    def __init__(self, model_name, vocab_size, config):

        super(LightningModel, self).__init__()

        self.config = config
        
        
        if model_name == 'lstm':
            self.model = LSTMClassifier(
                batch_size=config['data']['batch_size'],
                output_size=config['data']['num_classes'],
                hidden_size=config['model']['hidden_size'],
                vocab_size=vocab_size,
                embedding_length=config['model']['hidden_size'],
            )
        else:
            self.model = LSTMClassifier(
                batch_size=config['data']['batch_size'],
                output_size=config['data']['num_classes'],
                hidden_size=config['model']['hidden_size'],
                vocab_size=vocab_size,
                embedding_length=config['model']['hidden_size'],
            )
            


    def forward(self, input_ids, attention_mask=None):
        logits  = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return logits

    def configure_optimizers(self):
        return optim.AdamW(params=self.parameters(), lr=self.config['training']['lr'])

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target'].squeeze()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=-1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=-1).cpu(), average=self.config['training']['average'])
        wandb.log({"loss":loss, "accuracy":acc, "f1":f1})
        return {"loss":loss, "accuracy":acc, "f1":f1}

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target'].squeeze()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=-1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=-1).cpu(), average=self.config['training']['average'])
        wandb.log({"val_loss":loss, "val_accuracy":torch.tensor([acc]), "val_f1":torch.tensor([f1])})
        return {"val_loss":loss, "val_accuracy":torch.tensor([acc]), "val_f1":torch.tensor([f1])}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['val_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['val_f1'] for x in outputs]).mean()
        return {"val_loss":avg_loss, "val_accuracy":avg_acc, "val_f1":avg_f1,}

    def test_step(self, batch, batch_idx):
        input_ids, attention_mask, targets = batch['input_ids'], batch['attention_mask'], batch['target'].squeeze()
        logits = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = F.cross_entropy(logits, targets)
        acc = accuracy_score(targets.cpu(), logits.argmax(dim=-1).cpu())
        f1 = f1_score(targets.cpu(), logits.argmax(dim=-1).cpu(), average=self.config['training']['average'])
        return {"test_loss":loss, "test_accuracy":torch.tensor([acc]), "test_f1":torch.tensor([f1])}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        avg_acc = torch.stack([x['test_accuracy'] for x in outputs]).mean()
        avg_f1 = torch.stack([x['test_f1'] for x in outputs]).mean()
        return {"test_loss":avg_loss, "test_accuracy":avg_acc, "test_f1":avg_f1}