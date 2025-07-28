# CAER-S
 [![License: MIT](https://img.shields.io/badge/License-MIT-red.svg)](https://opensource.org/licenses/MIT)
 [![Open In Kaggle](https://img.shields.io/badge/Open%20in-Kaggle-blue)](https://www.kaggle.com/code/deadwish1/caer-model)



## Folder Structure
This project was created with [Pytorch-template](https://github.com/victoresque/pytorch-template) by Victor Huang. It has the following structure
  ```
  CAER/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  │
  ├── configs/ - holds configuration for training and testing
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── base/ - abstract base classes
  │   ├── base_data_loader.py
  │   ├── base_model.py
  │   └── base_trainer.py
  │
  ├── data_loader/ - anything about data loading goes here
  │   └── data_loaders.py
  │
  ├── data/ - default directory for storing input data
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py
  │   ├── metric.py
  │   └── loss.py
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── trainer/ - trainers
  │   └── trainer.py
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - small utility functions
      ├── util.py
      └── ...
  ```
Download [CAER-S dataset](https://caer-dataset.github.io/download.html)

## Usage 
Move into `configs` directory and create configuration file for training and/or testing:
```
{
  "name": "CAERS_Session",
  "n_gpu": 1,
  "arch": {
    "type": "CAERSNet",
    "args": {
      "num_classes": 7,
      "face_model_path": "/kaggle/working/FER_trained_model.pt",
      "body_backbone": "swin_t",
      "context_backbone": "resnet50_places",
      "freeze_backbones": true,
      "attention": true
    }
  },
  "loss": "cross_entropy",
  "metrics": [
    "accuracy"
  ],
  "test_loader": {
    "type": "CAERSDataLoader",
    "args": {
      "root": "/kaggle/input/caer-s/CAER-S/test",
      "detect_file": "/kaggle/working/test_with_body.txt",
      "train": false,
      "batch_size": 64,
      "shuffle": false,
      "num_workers": 2
    }
  },
  "train_loader": {
    "type": "CAERSDataLoader",
    "args": {
      "root": "/kaggle/input/caer-s/CAER-S/train",
      "detect_file": "/kaggle/working/train_with_body.txt",
      "batch_size": 64,
      "shuffle": true,
      "num_workers": 2
    }
  },
  "val_loader": {
    "type": "CAERSDataLoader",
    "args": {
      "root": "/kaggle/input/caer-s/CAER-S/test",
      "detect_file": "/kaggle/working/val_with_body.txt",
      "train": false,
      "batch_size": 64,
      "shuffle": false,
      "num_workers": 2
    }
  },
  "optimizer": {
      "type": "AdamW",
      "args": {
        "lr": 0.001,
        "weight_decay": 0.01,
        "betas": [0.9, 0.999]
      }
  },
  "lr_scheduler": {  
    "type": "StepLR",
    "args": {
      "step_size": 10,
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 50,
    "save_dir": "saved/",
    "save_period": 5,
    "verbosity": 2,
    "monitor": "max val_accuracy",
    "early_stop": 5,
    "tensorboard": false
  }
}
```

Once you've finished configuration, enter this snippet to train the model
```
python train.py --config [train config path]
```
To resume training from earlier checkpoint, add `--resume` flag
```
python train.py --resume [your checkpoint path]
```
To evaluate the model on test data, simply enter
```
import os

# Path to the log
log_file = 'output.log'
best_model_path = ''

try:
    with open(log_file, 'r') as f:
        # Lọc ra các dòng không rỗng
        lines = [line.strip() for line in f if line.strip()]
        if lines:
            best_model_path = lines[-1] # Get the last line
except FileNotFoundError:
    print(f"Error: Log file {log_file} not found.")


if best_model_path and 'model_best.pth' in best_model_path and os.path.exists(best_model_path):
    print(f"--- Best model found: {best_model_path} ---")
    print("--- Starting evaluation ---")
    
    # Replace this with your actual config path
    config_file = "/kaggle/working/CAER-S/CAER/configs/config.json" 
    
    !python -u test.py --config "{config_file}" --resume "{best_model_path}"
    
else:
    print(f"Error: No valid checkpoint path found in the file log")
    print(f"Last line read: '{best_model_path}'")
```

## Acknowledgements
Many thanks to Victor Huang and Khanh Nguyen for an amazing [Pytorch-template](https://github.com/victoresque/pytorch-template) , [Pytorch-CAER](https://github.com/ndkhanh360/CAER).
