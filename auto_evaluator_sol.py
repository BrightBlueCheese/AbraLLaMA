import sys
import os
import re
import pandas as pd
import numpy as np

import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.profilers import PyTorchProfiler
from lightning.pytorch.callbacks import ModelCheckpoint
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now you can import from models_mtr
# from chemllama_mtr import ChemLlama
import chemllama_mtr

# from .datamodule_finetune import CustomFinetuneDataModule
import datamodule_finetune_sol
# from .model_finetune import CustomFinetuneModel
import model_finetune_sol
import utils_sol

def auto_evaluator_level_2_sol(
    dir_model_mtr,
    # dir_model_mtr_ep_to_save:str,
    dir_model_ft_to_save:str,
    tokenizer,
    max_length:int,
    # molnet_dict:dict,
    # list_dataset_to_finetune:list,
    solute_or_solvent:str,
    num_workers:int,
    batch_size_pair=[32, 48],
    lr=0.0001,
    overwrite_level_2:bool=False,
    epochs:int=7,
    use_freeze:bool=True,
    validation:bool=True,
):

    """
    Evaluate the "one" pretrained MTR model through multiple finetuning benchmarking dataset.

    Parameters:
    # - dir_model_mtr_ep_to_save (str): The pretrained model for MTR with epoch.
    #                                    EX with 0 epoch:
    #                                    /master_dicrectory/pre_trained_model_MTR_name/model_MTR_with_epoch
    - batch_size_pair: The pair of the train and valid(+test) batch size (e.g. [32, 48] which is [32, int(32*1.5)])
    - overwrite_level_2 (bool): If there exists such folder that has the same "dir_model_mtr_ep_to_save", overwite it.
                                Warning! This option is only for "dir_model_mtr_ep_to_save". It's sub directory and files will be overwritten!
    """
    
    
    assert not (os.path.exists(dir_model_ft_to_save) and overwrite_level_2 == False), f"You sat 'overwrite_level_2' False and '{dir_model_ft_to_save}' already exists. Check it again."


    model_mtr = chemllama_mtr.ChemLlama.load_from_checkpoint(dir_model_mtr)
        
    # # local_dataset_to_finetune is a key of molnet_dict
    # list_local_finetuned_result = list()
    # for local_dataset_to_finetune in list_dataset_to_finetune:
        
    # dataset_dict = molnet_dict[local_dataset_to_finetune]
    # dataset_dict["dataset_name"] = local_dataset_to_finetune
    
    # dir_model_ft = f"{dir_model_mtr_ep_to_save}/{dataset_dict['dataset_name']}"
    dir_model_ft = f"{dir_model_ft_to_save}"
    # name_model_ft = utils_sol.model_ft_namer(dataset_dict['dataset_name'])
    name_model_ft = f"SolLlama_{solute_or_solvent}"

    # array_level_1, model_ft, data_loader_test
    array_level_1 = auto_evaluator_level_1_sol(
        model_mtr=model_mtr, 
        dir_model_ft=dir_model_ft, 
        name_model_ft=name_model_ft, 
        # dataset_dict=dataset_dict, 
        solute_or_solvent=solute_or_solvent,
        tokenizer=tokenizer, 
        max_length=max_length,
        num_workers=num_workers,
        batch_size_pair=batch_size_pair,
        lr=lr,
        epochs=epochs, 
        use_freeze=use_freeze,
        validation=validation,
    )
    
    return array_level_1
    
        # list_local_finetuned_result.append(array_level_1)
        
    # array_level_2 = np.vstack(list_local_finetuned_result)
    # array_level_2 shaped (number of epochs x len(list_dataset_to_finetune), number of columns at the bottom)
    # dataset_name, task, RMSE, MAE, p_value mantissam, p_value exponent, epoch, loss, loss_ranking, metric_1_ranking
    
    # return array_level_2

def auto_evaluator_level_1_sol(
    model_mtr, 
    dir_model_ft:str, 
    name_model_ft:str, 
    # dataset_dict:dict, 
    solute_or_solvent:str,
    tokenizer, 
    max_length:int,
    num_workers:int, ##
    batch_size_pair=[32, 48],
    lr=0.0001,
    epochs:int=7,
    use_freeze:bool=True,
    validation:bool=True,
):

    """
    Automate the entire process including preparing "one" finetuning dataset + finetuing + evalulation.
    This is a step before the level 2 evaluate automation.

    Parameters:
    - model_mtr: The pretrained model for MTR.
    - dir_model_ft (str): The directory where the model to be stored.
    - name_model_ft (str): The name of the model for finetune to be titled.
                           An example of the directory of the fintuned model with 0 epoch:
                           {dir_folder}/{name_model_ft}_ep_000
    - batch_size_pair: The pair of the train and valid(+test) batch size (e.g. [32, 48] which is [32, int(32*1.5)])
    """
    
    csv_logger = CSVLogger(
        save_dir=dir_model_ft,
        name=name_model_ft,
        version=0,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',
        # filename=name_model_ft + '_vloss_{val_loss:.3f}_ep_{epoch:02d}',
        filename=name_model_ft + '_{epoch:02d}',
        every_n_epochs=1,
        save_top_k=-1,
        enable_version_counter=False, # keep the version == 0
        save_weights_only=True,
    )
    checkpoint_callback.FILE_EXTENSION = ".pt"
    
    # Load dataset for finetune
    batch_size_for_train = batch_size_pair[0]
    batch_size_for_valid = batch_size_pair[1]

    data_module = datamodule_finetune_sol.CustomFinetuneDataModule(
        solute_or_solvent=solute_or_solvent,
        tokenizer=tokenizer,
        max_seq_length=max_length,
        batch_size_train=batch_size_for_train,
        batch_size_valid=batch_size_for_valid,
        # num_device=int(config.NUM_DEVICE) * config.NUM_WORKERS_MULTIPLIER,
        num_device=num_workers,
        train_only=not validation,
    )
    data_module.prepare_data()
    data_module.setup()
    steps_per_epoch = len(data_module.train_dataloader())

    # Load model and optimizer for finetune
    learning_rate = lr
        
    model_ft = model_finetune_sol.CustomFinetuneModel(
        model_mtr=model_mtr,
        steps_per_epoch=steps_per_epoch,
        warmup_epochs=1,
        max_epochs=epochs,
        learning_rate=learning_rate,
        # dataset_dict=dataset_dict,
        use_freeze=use_freeze,
    )
    
    trainer = L.Trainer(
        default_root_dir=dir_model_ft,
        # profiler=profiler,
        logger=csv_logger,
        accelerator='auto',
        devices='auto',
        # accelerator='gpu',
        # devices=[0],
        min_epochs=1,
        max_epochs=epochs,
        precision=32,
        callbacks=[checkpoint_callback]
    )
    trainer.fit(model_ft, data_module)
    if validation == True:
        trainer.validate(model_ft, data_module)

    list_validation_loss = pd.read_csv(f"{dir_model_ft}/{name_model_ft}/version_0/metrics.csv", usecols=['val_loss'])['val_loss'].dropna().tolist()[:epochs]

    # class_model_ft = CustomFinetuneModel
    # Level 1 Automation - Evaulate the finetuned model through every epoch
    array_level_1 = auto_evaluator_level_1_sub_sol(
        class_model_ft=model_ft, 
        list_validation_loss=list_validation_loss, 
        dir_model_ft=dir_model_ft, 
        name_model_ft=name_model_ft, 
        data_module=data_module,
        # dataset_dict=dataset_dict,
        solute_or_solvent=solute_or_solvent,
        trainer=trainer
    )
    
    return array_level_1

def auto_evaluator_level_1_sub_sol(
    class_model_ft,
    list_validation_loss,
    dir_model_ft:str,
    name_model_ft:str,
    data_module,
    # dataset_dict:dict,
    solute_or_solvent:str,
    trainer,
):

    """
    Evaluate the finetuned model by a single finetuning dataset.

    Guides for some parameters:
    - model_mtr: The pretrained model for MTR.
    - dir_model_ft (str): The directory where the model to be stored.
    - name_model_ft (str): The name of the model for finetune to be titled.
                           An example of the directory of the fintuned model with 0 epoch:
                           {dir_folder}/{name_model_ft}_ep_000
    """
    
    array_loss_ranking = utils_sol.rank_value_sol(
        list_value=list_validation_loss, 
        # dataset_dict=dataset_dict,
        is_loss=True,
    )
    # ranking : lower the better. ranking starting from 0

    print("- Epoch starts from 0")
    print("=======================================")
    
    list_level_1 = list()
    for ep in range(len(list_validation_loss)):

        local_model_ft = utils_sol.load_model_ft_with_epoch(
            class_model_ft=class_model_ft, 
            target_epoch=ep,
            dir_model_ft=dir_model_ft,
            name_model_ft=name_model_ft
        )
        
        result = trainer.predict(local_model_ft, data_module)
        result_pred = list()
        result_label = list()
        for bat in range(len(result)):
            result_pred.append(result[bat][0].squeeze())
            result_label.append(result[bat][1])

        list_local_model_ft_result = utils_sol.model_evalulator_sol(
            array_predictions=np.vstack(result_pred),
            array_labels=np.vstack(result_label),
            # dataset_dict=dataset_dict, 
            solute_or_solvent=solute_or_solvent,
            show_plot=False,
            print_result=False,
        )
        # dataset_name, task, RMSE, MAE, p_value mantissam, p_value exponent
        
        # add epoch (starting from 0) to the right 
        list_local_model_ft_result.append(ep)
        # dataset_name, task, metric1 (RMSE or ROC-AUC), metric2 (MAE or None), p_value mantissam, p_value exponent, epoch
        
        list_level_1.append(list_local_model_ft_result)
    print("=======================================")
    print("=======================================")

    # to get the metric_1 ranking
    array_level_1 = np.array(list_level_1)
    array_metric_1 = array_level_1[:, 2].astype('float32')
    array_metric_1_ranking = utils_sol.rank_value_sol(list_value=array_metric_1,
                                              # dataset_dict=dataset_dict,
                                              is_loss=False)

    # add loss, and ranking of the loss value to the right
    # reg: lower the better, class: higher the better
    array_level_1 = np.hstack((list_level_1, 
                               np.expand_dims(list_validation_loss, axis=1), 
                               np.expand_dims(array_loss_ranking, axis=1),
                               np.expand_dims(array_metric_1_ranking, axis=1))) 
    # solute_or_solvent, RMSE, MAE, p_value mantissam, p_value exponent, epoch, loss, loss_ranking, metric_1_ranking
    
    return array_level_1 
    #################################### EX #########################################
    # list_column_names = ['solute_or_solvent', 
    #                      'metric_1', 
    #                      'metric_2', 
    #                      'p_value_mantissa', 
    #                      'p_value_exponent', 
    #                      'epoch', 
    #                      'loss',
    #                      'loss_ranking',
    #                      'metric_1_ranking']
    # df_evaluation_level_1 = pd.DataFrame(array_level_1, columns=list_column_names)
    #################################################################################