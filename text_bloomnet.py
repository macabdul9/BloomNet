from config import config
from dataset.loader import get_loaders
from models.BloomNet import POSModel, NERModel, BloomNetClassifier
from transformers import AutoModel, AutoTokenizer, AutoModelForTokenClassification



if __name__ == '__main__':

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path="roberta-base")
    loaders = get_loaders(tokenizer=tokenizer, config=config['data'])
    batch = next(iter(loaders['fold0']['train']))

    text = batch['text']

    # pos = POSModel()

    # ner = NERModel()


    # pos_out = pos.forward(
    #     input_ids=batch['input_ids'],
    #     attention_mask=batch['attention_mask']
    # )


    # ner_out = pos.forward(
    #     input_ids=batch['input_ids'],
    #     attention_mask=batch['attention_mask']
    # )

    model = BloomNetClassifier()


    logits = model.forward(
        input_ids=batch['input_ids'],
        attention_mask=batch['attention_mask']
    )

    print(f'logits.shape = {logits.shape}')

    # print(f'pos_out.shape = {pos_out.shape} | ner_out.shape = {ner_out.shape}')