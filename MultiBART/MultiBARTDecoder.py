from transformers.models.bart.modeling_bart import BartDecoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch

import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import BartTokenizerFast

tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

class MultiBARTDecoder(BartDecoder):

    def __init__(self, config, embed_tokens: Optional[nn.Embedding] = None, target_len=128):
        config.decoder_ffn_dim = 3072
        self.target_len = target_len
        config.decoder_layers = 6
        super(MultiBARTDecoder, self).__init__(config, embed_tokens)

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        encoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, BaseModelOutputWithPastAndCrossAttentions]:

        #input_ids :: 'target' : batch size x (target_len(128) * num_decoders)
        if input_ids != None:
            batch_size, target_num_dec = input_ids.size()
            num_decoders = target_num_dec // self.target_len

            input_ids = torch.reshape(input_ids, (batch_size * num_decoders, self.target_len))
            attention_mask = torch.reshape(attention_mask, (batch_size * num_decoders, self.target_len))

        output = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                head_mask=head_mask,
                cross_attn_head_mask=cross_attn_head_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
        )

           
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_output,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            cross_attentions=output.cross_attentions,
        )


        
