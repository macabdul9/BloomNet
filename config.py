config = {
    "data":{
        "Text":"Text",
        "Label":"Label",
        "num_classes":6,
        "max_len":64,
        "batch_size":4,
        "num_workers":4,
        "folds":5,
    },
    
    "model":{
        "hidden_size":768
    },

    "training":{
        "epochs":20,
        "lr":2e-5, 
        "average":"macro",
    },
    
    "callback":{
        "monitor":"val_f1",
        "min_delta":0.001,
        "patience":5,
        "precision":32,
        "project":"bloom-taxonomy-classification",
    }


}