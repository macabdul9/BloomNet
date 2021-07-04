from models.hyfi.models import Model
import models.hyfi.constants as cs
import argparse
from transformers import AutoTokenizer



def config_parser(parser):
    # Data options
    parser.add_argument("--data", default=True, type=str, help="Data path.")

    # Sentence-level context parameters
    parser.add_argument("--men_nonlin", default="tanh", type=str, help="Non-linearity in mention encoder")
    parser.add_argument("--ctx_nonlin", default="tanh", type=str, help="Non-linearity in context encoder")
    parser.add_argument("--num_layers", default=1, type=int, help="Number of layers in MobiusGRU")
    parser.add_argument("--space_dims", default=20, type=int, help="Space dims.")

    # Component metrics
    parser.add_argument("--embedding_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")
    parser.add_argument("--encoder_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")
    parser.add_argument("--attn_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")
    parser.add_argument("--concat_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")
    parser.add_argument("--mlr_metric", default=cs.HY, type=str, help="hyperbolic | euclidean")

    # Other parameters
    parser.add_argument("--input_dropout", default=0.3, type=float, help="Dropout over input.")
    parser.add_argument("--concat_dropout", default=0.2, type=float, help="Dropout in concat.")
    parser.add_argument("--classif_dropout", default=0.0, type=float, help="Dropout in classifier.")
    parser.add_argument("--crowd_cycles", default=5, type=int, help="Number of crowd re-train.")
    parser.add_argument("--learning_rate", default=0.0005, type=float, help="Starting learning rate.")
    parser.add_argument("--weight_decay", default=0.00, type=float, help="L2 Regularization.")
    parser.add_argument("--batch_size", default=900, type=int, help="Batch size.")
    parser.add_argument("--epochs", default=30, type=int, help="Number of training epochs.")
    parser.add_argument("--max_grad_norm", default=5, type=float,
                        help="If the norm of the gradient vector exceeds this, renormalize it to max_grad_norm")
    parser.add_argument("--patience", default=50, type=int, help="Patience for lr scheduler")
    parser.add_argument("--export_path", default="", type=str, help="Name of model to export")
    parser.add_argument("--export_epochs", default=20, type=int, help="Export every n epochs")
    parser.add_argument("--log_epochs", default=4, type=int, help="Log examples every n epochs")
    parser.add_argument("--load_model", default="", type=str, help="Path of model to load")
    parser.add_argument("--train_word_embeds", default=0, type=int, help="Wether to train word embeds or not")
    parser.add_argument("--seed", default=-1, type=int, help="Seed")
    parser.add_argument("--c", default=1.0, type=float, help="c param to project embeddings")
    parser.add_argument("--attn", default="softmax", type=str, help="Options: sigmoid | softmax")



if __name__ == '__main__':


    parser = argparse.ArgumentParser("train.py")
    config_parser(parser)
    args = parser.parse_args()


    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")




    # model = Model()
    print(tokenizer.vocab)
    print(cs.TYPE_VOCAB)