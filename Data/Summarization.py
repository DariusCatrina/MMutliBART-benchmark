from datasets import load_dataset
from transformers import BartTokenizer, BartTokenizerFast

import torch
from torch.utils.data import Dataset

from Data.data import BackboneDataset



class Summarization_Wrapper(BackboneDataset):

    def __init__(self, dataset_name : str, split: str, bart_seq_len : int, seq_len: int, fast_tokenizer: bool, tensor_type: str, num_decoders: int):
        super().__init__(dataset_name=dataset_name, 
                        split=split, 
                        seq_len=seq_len,
                        fast_tokenizer = fast_tokenizer, 
                        tensor_type=tensor_type,
                        bart_seq_len=bart_seq_len)
        
        self.num_decoders = num_decoders
        self.target_len = 128



    def chunknize(self, tokenized_input, num_chunks, len_per_chunk, pad_id, target=False):
        prev_chunk_len = 0
        input_ids = torch.zeros((num_chunks, len_per_chunk), dtype=torch.long)
        attention_mask = torch.zeros((num_chunks, len_per_chunk), dtype=torch.long)
        loss_mask = torch.zeros((num_chunks, len_per_chunk), dtype=torch.long)
        ignored_tokens = num_chunks * len_per_chunk

        input = tokenized_input['input_ids'][0]


        for i in range(0, num_chunks):
          if target==True :
            id = 'Chunk: ' + str(i) + ' '  
          else:
            id = ''

          prefix = self.tokenizer(id, add_special_tokens=False,
                                      return_attention_mask=False, 
                                      return_token_type_ids=False, 
                                      return_tensors=self.tensor_type)['input_ids'][0]
          
          chunked_context = input[prev_chunk_len:(prev_chunk_len + (len_per_chunk - 2 - prefix.size(dim=0)))]
          attention_length = None

          if chunked_context.size(dim=0) == (len_per_chunk - 2 - prefix.size(dim=0)): #Full size chunk
              attention_length = chunked_context.size(dim=0) + 2
              input_ids[i] = torch.cat((self.bos_id, prefix, chunked_context, self.eos_id), dim=0)
              
          elif chunked_context.size(dim=0) == 0: #Emplty chunk
              attention_length = 0
              padding = torch.full((1, len_per_chunk - prefix.size(dim=0)), pad_id)[0]
              input_ids[i] = torch.cat((prefix, padding), dim=0)

          else:#Chunk with part context, part padding 
              padd_dim = (len_per_chunk - 2 - prefix.size(dim=0)) - chunked_context.size(dim=0)
              padding_tensor = torch.full((1, padd_dim), pad_id)[0]
              attention_length = chunked_context.size(dim=0) + 2 - padd_dim
              input_ids[i] = torch.cat((self.bos_id, prefix, chunked_context, self.eos_id, padding_tensor), dim=0)
              
          prev_chunk_len = prev_chunk_len + len_per_chunk - 2
          attention_mask[i][:attention_length] = 1
          if target==True:
            ignored_tokens -= prefix.size(dim=0)
            loss_mask[i][prefix.size(dim=0):attention_length] = 1

        if target:
            return (input_ids, attention_mask, loss_mask, ignored_tokens)
        return (input_ids, attention_mask)

    def __getitem__(self, idx):
        
        if self.seq_len %self.bart_seq_len != 0: #check if the ratio between the sequence length and bart's sequence length is a float or an int
            num_chunks = int(self.seq_len /self.bart_seq_len) + 1
        else:
            num_chunks = int(self.seq_len /self.bart_seq_len)


        tokenized_input = self.tokenizer(self.dataset[idx]['input'],add_special_tokens=False, 
                                                           return_attention_mask=False, 
                                                           return_token_type_ids=False, 
                                                           return_tensors=self.tensor_type)

        input_ids, attention_mask = self.chunknize(tokenized_input, num_chunks,self.bart_seq_len, self.pad_id)
        

        if(self.dataset['output'][idx] != None):
            target = self.tokenizer(self.dataset[idx]['output'],add_special_tokens=False,
                                                            return_attention_mask=False, 
                                                            return_token_type_ids=False, 
                                                            return_tensors=self.tensor_type)
            decoder_input_ids, target_attention_mask, loss_mask, ignored_tokens = self.chunknize(target, self.num_decoders, self.target_len, self.pad_id, target=True)
    
            return_dict = {
                    'input_ids' : input_ids.reshape(num_chunks*self.bart_seq_len).long(),
                    'attention_mask' : attention_mask.reshape(num_chunks*self.bart_seq_len).long(),
                    'labels': decoder_input_ids.reshape(self.num_decoders*self.target_len).long(),
                    'decoder_attention_mask': target_attention_mask.reshape(self.num_decoders*self.target_len).long(),
                    'loss_mask' : loss_mask.reshape(self.num_decoders*self.target_len).long(),
                    'num_tokens_ignored' : torch.tensor(ignored_tokens).long()
                    #'raw_target': self.dataset[idx]['output']
                }
        else:
          return_dict = {
                    'input_ids' : input_ids.reshape(num_chunks*self.bart_seq_len).long(),
                    'attention_mask' : attention_mask.reshape(num_chunks*self.bart_seq_len).long(),
                }



        return return_dict
 

