# fashion-mnist-pytorch

This project trains a PyTorch model to classify images from the fashion mnist dataset ([link](https://github.com/zalandoresearch/fashion-mnist)). The dataset comprises fo 60,000 grayscale images of size 28x28, each corresponding to one of 10 classes.

The model is a Convolutional Neural Network (CNN) with customisable parameters. 

<img width="800" alt="Screenshot 2023-08-24 at 17 49 07" src="https://github.com/samuelcortinhas/fashion-mnist-pytorch/assets/128174954/6304de39-3061-4036-844f-f4c72eee7e5c">

# How to use

1. (Optional) Modify the `config.json` file with your desired parameters. 
2. Run `python src/train.py` in the command line to train a new model. 

# Documentation

The following are descriptions of the parameters in `config.json`.

* `seed` - random seed, e.g. 0
* `batch_size` - batch size, e.g. 64
* `learning_rate` - initial learning rate, e.g. 0.001
* `n_epochs` - number of training epochs, e.g. 10
* `conv1_channels` - number of channels in first convolutional layer, e.g. 64
* `conv2_channels` - number of channels in second convolutional layer, e.g. 128
* `linear1_size` - number of neurons in first linear layer, e.g. 256
* `linear2_size` - number of neurons in second linear layer, e.g. 128
* `dropout_rate` - dropout rate, e.g. 0.25
* `verbose` - (bool) print metrics at each iteration, e.g. true
* `train_path` - path to training dataset, e.g. "data/fashion-mnist_train.csv"
* `test_path` - path to test dataset, e.g. "data/fashion-mnist_test.csv"
* `debug` - (bool) whether to run in debug mode (small sample of datasets), e.g. false
* `logging` - (bool) whether to enable W&B experiment tracking, e.g. true
* `save_model` - (bool) whether to save the trained model, e.g. true
* `save_model_name` - name of trained model to save, e.g. "trained_convnet_v1"
* `inference_model_name` - name of saved model to use for inference, e.g. "trained_convnet_v2"

# Experiment tracking

(Optional) If you would like to track your experiments with Weights and Biases, then do the following:
* Set `logging` to true in `config.json`
* Create a `.env` file and set values for the variables `WANDB_API_KEY` and `WANDB_ENTITY`. The former is your personal api key that can found at `https://wandb.ai/authorize` and the later will be your account name. For example:

```python
WANDB_API_KEY=0123456789
WANDB_ENTITY=johnsmith
```

This will create a dashboard for your experiments, which can be customised. Below is an example. 

<img width="800" alt="Screenshot 2023-08-24 at 18 10 34" src="https://github.com/samuelcortinhas/fashion-mnist-pytorch/assets/128174954/7afd3210-c2f4-496d-bcc3-849f135bbfe2">
