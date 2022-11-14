from torch.utils.data import DataLoader
import json
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

class MMultiBART_Trainer(Seq2SeqTrainer):
    train_dataloader : DataLoader = None
    validation_dataloader : DataLoader = None

    def get_train_dataloader(self) -> DataLoader:
        if MMultiBART_Trainer.train_dataloader == None:
            raise "Invalid/Null training dataloader"
        return MMultiBART_Trainer.train_dataloader

    def get_eval_dataloader(self, eval_dataset = None) -> DataLoader:
        if MMultiBART_Trainer.validation_dataloader == None:
            raise "Invalid/Null evaluation dataloader"
        return MMultiBART_Trainer.validation_dataloader

    # def compute_loss(self, model, inputs, return_outputs=False):
    #         return model(input_ids=inputs['input_ids'],
    #                  attention_mask=inputs['attention_mask'],
    #                  labels=inputs['labels'],
    #                  decoder_attention_mask=inputs['decoder_attention_mask']).loss
    #     return None


def populate_config_file(json_config='training_args.json'):
    import dataclasses

    with open(json_config) as file: 
        config_dict = json.load(file)

    keys = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
    values = {k: v for k, v in config_dict.items() if k in keys}


    return Seq2SeqTrainingArguments(**values)

         





