# The Role of Model Architecture and Scale in Predicting Molecular Properties: Insights from Fine-Tuning RoBERTa, BART, and LLaMA

This code was built based on our previous project [The Role of Model Architecture and Scale in Predicting Molecular Properties: Insights from Fine-Tuning RoBERTa, BART, and LLaMA](https://arxiv.org/abs/2405.00949). 


## Requirements

To install requirements:

```setup
pip install -r requirements_cuda118.txt
```

>📋  The experiments were done under CUDA 11.8

## Dataset
1. ```./dataset_ft/Abraham***_cleared.csv``` already preprocessed.


## Training
To train the model(s) in the paper, move to `AbraLLaMA` (the main directory) and run:

```fine-tuning AbraLLaMA
python run_auto_llama.py
```

## Evaluation

To check the model's metrics, loss, and etc., move to `AbraLLaMA/evaluations` (the main directory):

metric_1(RMSE), metric_2/loss(MAE)



## Pre-trained Models

We have used one of the pretrained ChemLLaMA-MTR model from our [previous project](https://github.com/BrightBlueCheese/transformers_and_chemistry/tree/main)
```pretrained-mtr
./model_mtr/ChemLlama_Medium_30m_vloss_val_loss=0.029_ep_epoch=04.ckpt
```

## Demo Run
You can also train AbraLLaMA demo version wiht Jupyter
1. Open ```run_demo.ipynb```


## Contributing

>📋  MIT

## Authors' Note
Please use this code only for social goods and positive impact.