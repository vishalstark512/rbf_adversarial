# -*- coding: utf-8 -*-
"""Model config in json format"""

CFG = {
    "data": {
        "path": "oxford_iiit_pet:3.*.*",
        "image_szie": 256,
        "load_with_info": True
    },

    "train": {
        "batch_size": 8,
        "buffer_size": 1000,
        "epochs": 20,
        "val_subsplits": 5,
        "optimizer": {
            "type": "adam"
        },
        "metrics": ["accuracy"]
    },

    "model": {
        "input": [64, 64, 1],
        "filter": {
            "l1": 16,
            "l2": 32,
            "l3": 64,
            "l4": 128,
            "l5": 256,
            "kernels": 3
        },
        "output": 1
    }
}