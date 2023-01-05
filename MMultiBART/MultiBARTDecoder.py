from transformers.models.bart.modeling_bart import BartDecoder
from transformers.modeling_outputs import BaseModelOutputWithPastAndCrossAttentions
import torch

import torch.nn as nn
from typing import List, Optional, Tuple, Union

from transformers import BartTokenizerFast

import math

tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


class MultiBARTDecoder(BartDecoder):

    def __init__(self, config, embed_tokens: Optional[nn.Embedding] = None, decoder_len=128, num_decoders=4, do_generate=False):
        config.decoder_ffn_dim = 3072
        self.decoder_len = decoder_len
        self.num_decoders = num_decoders
        self.do_generate = do_generate
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

        if self.do_generate:
            return super().forward(
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
        

        if input_ids != None:
            batch_size, _ = input_ids.size()  # batch_size x dec_len*num_decoders
            _, seq_len, hidden_size = encoder_hidden_states.size()

            #encoded_ids: batch_size x seq_len x hidden_size: 5 x (4*128), 5 x 4096 x 768
            # decoder_output = torch.zeros(batch_size, self.num_decoders*self.decoder_len, 768).to(device)

            input_ids = torch.reshape(input_ids, (batch_size * self.num_decoders, self.decoder_len))
            attention_mask = torch.reshape(attention_mask, (batch_size *self.num_decoders, self.decoder_len))
            encoder_hidden_states = torch.reshape(encoder_hidden_states, (batch_size *self.num_decoders, seq_len//self.num_decoders, hidden_size))
            encoder_attention_mask = torch.reshape(encoder_attention_mask, (batch_size *self.num_decoders, seq_len//self.num_decoders))

            #[2, 4, 128] -> [8, 128]
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
                        return_dict=return_dict)

            decoder_output = torch.reshape(output.last_hidden_state, (batch_size, self.num_decoders*self.decoder_len, 768)).to(device)
            # for i in range(self.num_decoders):
            #     output = super().forward(
            #             input_ids=input_ids[:,i,:],
            #             attention_mask=attention_mask[:,i,:],
            #             encoder_hidden_states=encoder_hidden_states,
            #             encoder_attention_mask=encoder_attention_mask,
            #             head_mask=head_mask,
            #             cross_attn_head_mask=cross_attn_head_mask,
            #             past_key_values=past_key_values,
            #             inputs_embeds=inputs_embeds,
            #             use_cache=use_cache,
            #             output_attentions=output_attentions,
            #             output_hidden_states=output_hidden_states,
            #             return_dict=return_dict,
            #     ) 
            #     decoder_output[:,i:i+self.decoder_len, :] = output.last_hidden_state

           
        return BaseModelOutputWithPastAndCrossAttentions(
            last_hidden_state=decoder_output,
            past_key_values=output.past_key_values,
            hidden_states=output.hidden_states,
            attentions=output.attentions,
            cross_attentions=output.cross_attentions,
        )


        
