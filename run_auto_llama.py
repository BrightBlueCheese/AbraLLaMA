import sys
import os

# Since FT dataset is small, using one GPU should be proper.
# os.environ["CUDA_VISIBLE_DEVICES"]= "0"


import torch
import numpy as np
import pandas as pd
import warnings
import lightning as L
torch.set_float32_matmul_precision('high')

# Filter out FutureWarning and UnderReviewWarning messages from pl_bolts
warnings.filterwarnings("ignore", module="pl_bolts")

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import tokenizer_sol
import auto_evaluator_sol
import utils_sol

print(os.path.dirname(__file__))


torch.manual_seed(1004)
np.random.seed(1004)

print(os.getcwd())

"""
Using learning rate bigger than the default setting is not that recommanded since we don't freeze the MTR model.
But lower lr could work.

Be aware of doing version control (ver_ft). Make sure you keep the same version for both 'solute' and 'solvent' otherwise, you will get confused.

The variable "dir_model_ft_to_save" is where the FT model get saved.
The result csv files will be located at 'evaluations/corresponding version/solute and (or) solvent.csv'

You can run this code by

python run_auto_llama.py

But makes sure you are running this in your virtual environment that all requirements_cuda118.txt installed
"""


"""
# You can run both 'solute' and 'solvent' at one run by doing the below
for solute_or_solvent in ['solute' ,'solvent']:
    The REST of the codes except the variant solute_or_solvent right below with this (SAME) indentation levels
"""
# Clone the pretrained model's repository
utils_sol.get_pretrained_model()

#### Hyper Parameters ##### <- You can control these parameters as you want
# solute_or_solvent = 'solvent'
solute_or_solvent = 'solute'
ver_ft = 0 # version control for FT model & evaluation data # Or it will overwrite the models and results
batch_size_pair = [64, 64] if solute_or_solvent == 'solute' else [10, 10] # [train, valid(test)] 
# since 'solute' has very small dataset. So I thinl 10 for train and 10 for valid(test) should be the maximum values.
lr = 0.0001 
epochs = 20
use_freeze = False  # Freeze the model or not # False measn not freezing
overwrite_level_2 = True # If you don't want to overwrite the models and csv files, then change this to False
###########################


# I just reused our previous research code with some modifications.
dir_main = "./" # or Parent Dir
name_model_mtr = "ChemLlama_Medium_30m_vloss_val_loss=0.029_ep_epoch=04.ckpt" 

dir_model_mtr = f"{dir_main}/model_mtr/{name_model_mtr}"

max_seq_length = 512

tokenizer = tokenizer_sol.fn_load_tokenizer_llama(
    max_seq_length=max_seq_length,
)
max_length = max_seq_length
num_workers = 2

dir_model_ft_to_save = f"{dir_main}/save_models_ft/ft_version_{ver_ft}"

array_level_2 = auto_evaluator_sol.auto_evaluator_level_2_sol(
    dir_model_mtr=dir_model_mtr,
    dir_model_ft_to_save=dir_model_ft_to_save,
    tokenizer=tokenizer,
    max_length=max_seq_length,
    solute_or_solvent=solute_or_solvent,
    num_workers=num_workers,
    batch_size_pair=batch_size_pair,
    lr=lr,
    overwrite_level_2=overwrite_level_2,
    epochs=epochs,
    use_freeze=use_freeze,
)

print(array_level_2.shape)
print(array_level_2)

list_column_names_level_2 = [
    'solute_or_solvent', 
    'metric_1', 
    'metric_2', 
    'epoch', 
    'loss',
    'loss_ranking',
    'metric_1_ranking'
]

df_evaluation_level_2 = pd.DataFrame(array_level_2, columns=list_column_names_level_2)

os.makedirs(f'{os.path.dirname(__file__)}/evaluations/ft_version_{ver_ft}', exist_ok=True)
df_evaluation_level_2.to_csv(f'{os.path.dirname(__file__)}/evaluations/ft_version_{ver_ft}/{solute_or_solvent}.csv', index=False)



