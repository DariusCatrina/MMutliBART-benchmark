from transformers import BartConfig
from transformers.models.bart.modeling_bart import BartEncoder, BartEncoderLayer
from transformers.modeling_outputs import BaseModelOutput
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')



class MultiBARTEncoder(BartEncoder):

    bart_seq_len = None
    def __init__(self, config, embed_tokens: Optional[nn.Embedding] = None):
        config.d_model = 768
        config.encoder_layers = 6
        config.encoder_ffn_dim = 3072

        super(MultiBARTEncoder, self).__init__(config, embed_tokens)
        

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutput]:


        batch_size, seq_len = input_ids.size() # batch x seq_len(1024,2048,4096 etc)
        num_chunks = seq_len // MultiBARTEncoder.bart_seq_len 
        #assert num_chunks == 1, (seq_len, num_chunks, MultiBARTEncoder.bart_seq_len) 

        # 1
        input_ids = torch.reshape(input_ids, (batch_size * num_chunks, MultiBARTEncoder.bart_seq_len))
        attention_mask = torch.reshape(attention_mask, (batch_size * num_chunks, MultiBARTEncoder.bart_seq_len))
        # print(input_ids.size())
        # print(attention_mask.size())

        encoder_outputs = super().forward(input_ids=input_ids, 
                                           attention_mask=attention_mask,
                                           head_mask=head_mask,
                                           inputs_embeds=inputs_embeds,
                                           output_attentions=output_attentions,
                                           output_hidden_states=output_hidden_states,
                                           return_dict=return_dict) # num_chunks X bart_seq_len X 768

        concat_embeddings = torch.reshape(encoder_outputs[0], (batch_size, num_chunks*MultiBARTEncoder.bart_seq_len, 768)) # batch_size x seq_len x embedding dim
        # print(concat_embeddings.size())
        return BaseModelOutput(
                last_hidden_state=concat_embeddings, hidden_states=encoder_outputs.hidden_states, attentions=encoder_outputs.attentions
            ) 