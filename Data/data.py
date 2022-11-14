from datasets import load_dataset
from transformers import BartTokenizer, BartTokenizerFast

import torch
from torch.utils.data import Dataset


class BackboneDataset(Dataset):
    dataset_provider: str = "tau/scrolls"
    bart_name: str = "facebook/bart-base"

    def __init__(self, dataset_name:str, split: str, seq_len: int, fast_tokenizer: bool, tensor_type: str = 'pt', bart_seq_len: int = 1024):
        '''
        Args:
            split : str, Dataset split (i.e. training/validation/test)
            seq_len: int, Input length that would normally be used for a long context model (i.e. 512, 1024, 2048, 3072, 4096)
            fast_tokenizer: bool(True/False), If the tokenizer is the base one or the fast one
            segmentation_type: int(1 or 2), 1 is for the hypotesis in each input chunk, 2 is for the hypotesis only in the first chunk
            tensor_type: str(tpu, jnp, np, pt), The type of tensor to be returned(Pytorch Tensor, JAX Tensor, etc)
        '''
        
        print(f'Downloading the dataset for {split} split...')
        self.dataset = load_dataset(path=BackboneDataset.dataset_provider, name=dataset_name, split=split)

        self.seq_len = seq_len #512, 1024, 2048, 3072, 4096

        print('Downlading the tokenizer...')
        if fast_tokenizer==True:
            self.tokenizer = BartTokenizerFast.from_pretrained(BackboneDataset.bart_name, add_prefix_space=True)
        else:
            self.tokenizer = BartTokenizer.from_pretrained(BackboneDataset.bart_name, add_prefix_space=True)
        


        self.bos_id = torch.tensor([self.tokenizer.convert_tokens_to_ids('<s>')]) #begining of sentance token id
        self.eos_id = torch.tensor([self.tokenizer.convert_tokens_to_ids('</s>')]) #end of sentance token id
        self.pad_id = torch.tensor(self.tokenizer.convert_tokens_to_ids('<pad>')) #padding token id

        self.tensor_type = tensor_type # tf, pt, np, jnp
        self.bart_seq_len = bart_seq_len

    def __len__(self) -> int:
        return self.dataset.num_rows