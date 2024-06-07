gaia = {
    # dataset
    "dataset": "gaia",
    "dataset_dir": "../datasets/new_gaia",
    "save_dir": "data/gaia",
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 10,
    "sample_interval": 60,
    "drain_config": {
        "drain_save_path": "data/gaia/drain.bin",
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # label info
    "failures": "login memory file access",
    "services": "webservice mobservice dbservice redisservice logservice",
    "instances": "webservice1 webservice2 logservice1 logservice2 mobservice1 mobservice2 redisservice1 redisservice2 dbservice1 dbservice2",
    "label_type": "failure_type",
    "num_class": 4,
    # cuda
    "use_cuda": False,
    "gpu": 0,
    # training
    "optim": "AdamW",
    "epochs": 50,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    # model
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 2,
    "dropout": 0.3,
}
aiops22 = {
    # dataset
    "dataset": "aiops22",
    "dataset_dir": "../datasets/aiops2022-pre",
    "save_dir": "data/aiops22",
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 10,
    "sample_interval": 60,
    "drain_config": {
        "drain_save_path": "data/aiops22/drain.bin",
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # label info
    "failures": "cpu io memory network process",
    "services": "adservice cartservice checkoutservice currencyservice emailservice frontend paymentservice productcatalogservice recommendationservice shippingservice",
    "instances": "adservice-0 adservice-1 adservice-2 adservice2-0 cartservice-0 cartservice-1 cartservice-2 cartservice2-0 checkoutservice-0 checkoutservice-1 checkoutservice-2 checkoutservice2-0 currencyservice-0 currencyservice-1 currencyservice-2 currencyservice2-0 emailservice-0 emailservice-1 emailservice-2 emailservice2-0 frontend-0 frontend-1 frontend-2 frontend2-0 paymentservice-0 paymentservice-1 paymentservice-2 paymentservice2-0 productcatalogservice-0 productcatalogservice-1 productcatalogservice-2 productcatalogservice2-0 recommendationservice-0 recommendationservice-1 recommendationservice-2 recommendationservice2-0 shippingservice-0 shippingservice-1 shippingservice-2 shippingservice2-0",
    "label_type": "failure_type",
    "num_class": 5,
    # cuda
    "use_cuda": False,
    "gpu": 0,
    # training
    "optim": "SGD",
    "epochs": 120,
    "lr": 0.001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    # model
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 1,
    "dropout": 0.35,
}
platform = {
    # dataset
    "dataset": "platform",
    "dataset_dir": "../datasets/new_platform",
    "save_dir": "data/platform",
    "log_name": "experiment",
    "use_tmp": True,
    "num_workers": 10,
    "sample_interval": 60,
    "drain_config": {
        "drain_save_path": "data/platform/drain.bin",
        "drain_config_path": "dataset/drain3/drain.ini",
    },
    "bert_config": {
        "tokenizer_path": "cache/bert-base-uncased",
        "model_path": "cache/bert-base-uncased",
    },
    # label info
    "failures": "cpu http/grpc memory network pod_failure",
    "services": "cartservice checkoutservice currencyservice emailservice frontend paymentservice productcatalogservice recommendationservice shippingservice",
    "instances": "cartservice checkoutservice currencyservice emailservice frontend paymentservice productcatalogservice recommendationservice shippingservice",
    "label_type": "failure_type",
    "num_class": 5,
    # cuda
    "use_cuda": False,
    "gpu": 0,
    # training
    "epochs": 50,
    "lr": 0.0001,
    "weight_decay": 0.0001,
    "batch_size": 32,
    # model
    "max_len": 512,
    "d_model": 768,
    "nhead": 8,
    "d_ff": 256,
    "layer_num": 1,
    "dropout": 0.1,
}

CONFIG_DICT = {"gaia": gaia, "aiops22": aiops22, "platform": platform}
