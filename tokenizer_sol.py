from transformers import LlamaTokenizerFast, RobertaTokenizerFast, BartTokenizerFast
import json
import os


def fn_load_tokenizer_llama(
    max_seq_length,
    dir_tokenizer: str = "./tokenizer.json",
    # dir_tokenizer:str = os.path.abspath(os.path.join(os.getcwd(), '..', "models_mtr/tokenizer.json")), # for JUP
    add_eos_token:bool = True,
):

    tokenizer = LlamaTokenizerFast(
        tokenizer_file=dir_tokenizer,
        model_max_length=max_seq_length,
        padding_side="right",
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        add_eos_token=add_eos_token,
    )
    tokenizer.add_special_tokens({"pad_token": "<pad>", "sep_token": "</s>", "cls_token": "<s>", "mask_token":"<mask>"})
    # tokenizer.add_special_tokens({"pad_token": "<pad>"})

    return tokenizer

def fn_load_descriptor_list(
    key_descriptor_list,
    dir_descriptor_list,
):

    with open(dir_descriptor_list, "r") as js:
        list_descriptor = json.load(js)[key_descriptor_list]

    return list_descriptor
