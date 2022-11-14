from transformers import BartForConditionalGeneration, BartConfig
from transformers.modeling_outputs import Seq2SeqLMOutput, BaseModelOutput
from transformers.models.bart.modeling_bart import shift_tokens_right
from typing import List, Optional, Tuple, Union

from MultiBART.MultiBARTEncoder import MultiBARTEncoder
from MultiBART.MultiBARTDecoder import MultiBARTDecoder


import torch
import torch.nn as nn
from torch.optim import AdamW



device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
bart_config = BartConfig()

class MultiBART(BartForConditionalGeneration):
    PRETRAINED_BART = BartForConditionalGeneration.from_pretrained("facebook/bart-base")

    def __init__(self):
        super(MultiBART, self).__init__(bart_config)       
        self.init_backbone()

        self.criterion = nn.CrossEntropyLoss()#reduction='none')
        self.to(device)
        self.check_dim = 1

        
    def set_number_of_decoders(self, decoder_chunks):
        self.number_of_decoders = number_of_decoders

    def get_encoder(self):
        return self.encoder
    
    def get_decoder(self):
        return self.decoder

    def init_backbone(self):
        self.encoder = MultiBARTEncoder(config=bart_config, embed_tokens=MultiBART.PRETRAINED_BART.model.shared)
        self.decoder = MultiBART.PRETRAINED_BART.get_decoder()
        self.lm_head = MultiBART.PRETRAINED_BART.get_output_embeddings()
        self.logits_bias = MultiBART.PRETRAINED_BART.final_logits_bias.to(device)
        self.logits_bias.requires_grad = True

    def apply_pretrained_weights(self, pretrained_model):
        self.encoder.load_state_dict(pretrained_model.get_encoder().state_dict())
        self.decoder.load_state_dict(pretrained_model.get_decoder().state_dict())


    # def generate_loss_attention_mask(self, target_len):
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        head_mask: Optional[torch.Tensor] = None,
        decoder_head_mask: Optional[torch.Tensor] = None,
        cross_attn_head_mask: Optional[torch.Tensor] = None,
        encoder_outputs: Optional[List[torch.FloatTensor]] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        decoder_inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Seq2SeqLMOutput]:

        if labels is not None:
            use_cache = False
            if decoder_input_ids is None and decoder_inputs_embeds is None:
                decoder_input_ids = shift_tokens_right(labels, self.config.pad_token_id, self.config.decoder_start_token_id)
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = shift_tokens_right(input_ids, self.config.pad_token_id, self.config.decoder_start_token_id)

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )

        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        #MultiBART - Seq2Seq
        if encoder_outputs is None:
            
            encoder_outputs = self.encoder(input_ids=input_ids, 
                                            attention_mask=attention_mask,
                                            head_mask=head_mask,
                                            inputs_embeds=inputs_embeds,
                                            output_attentions=output_attentions,
                                            output_hidden_states=output_hidden_states,
                                            return_dict=return_dict)
            
        decoder_outputs = self.decoder(input_ids=decoder_input_ids, 
                               attention_mask=decoder_attention_mask, 
                               encoder_hidden_states=encoder_outputs[0], 
                               encoder_attention_mask=attention_mask,
                               head_mask=decoder_head_mask,
                               cross_attn_head_mask=cross_attn_head_mask,
                               past_key_values=past_key_values,
                               inputs_embeds=decoder_inputs_embeds,
                               use_cache=use_cache,
                               output_attentions=output_attentions,
                               output_hidden_states=output_hidden_states,
                               return_dict=return_dict) # (batch_size) X target_max_size X 768 : [4, 128, 768]
        
        lm_logits = self.lm_head(decoder_outputs[0]) + self.logits_bias # [4, 128, 50265]
        if(self.check_dim == 1):
            print(lm_logits.size())
            self.check_dim = 0

        masked_lm_loss = None

        if labels is not None:
            masked_lm_loss = self.criterion(lm_logits.view(-1, self.config.vocab_size), labels.view(-1))# [512, 50k], [512] => batch_size
            

        return Seq2SeqLMOutput(
            loss=masked_lm_loss,
            logits=lm_logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )







