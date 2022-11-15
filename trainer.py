import torch
from torch.utils.data import DataLoader

from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)  


from transformers.trainer_utils import (
    EvalLoopOutput,
    EvalPrediction,
    denumpify_detensorize,
    has_length
)

import statistics
from tqdm import tqdm 
import json


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


    def evaluation_loop(self, dataloader: DataLoader, description: str, prediction_loss_only: bool = None, ignore_keys = None, metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        validation_dataset = getattr(dataloader, "dataset", None)
        
        self.all_predicitons = []
        self.all_losses = []

        print(f"***** Running {description} *****")
        num_examples = self.num_examples(dataloader)
        if has_length(dataloader):
            print(f"  Num examples = {num_examples}")
        else:
            print("  Num examples: Unknown")
        print(f"  Batch size = {self.args.eval_batch_size}")

        # if full fp16 or bf16 eval is wanted and this ``evaluation`` or ``predict`` isn't called
        # while ``train`` is running, cast it to the right dtype first and then put on device

        if not self.is_in_train:
            if self.args.fp16_full_eval:
                model = model.to(dtype=torch.float16, device=self.args.device)
            elif self.args.bf16_full_eval:
                model = model.to(dtype=torch.bfloat16, device=self.args.device)

        model.eval()
        model.zero_grad()

        dataset_tqdm = tqdm(dataloader)
        for step, batch in enumerate(dataset_tqdm):
            batch = self.to_device(batch)
            with torch.no_grad():
                loss = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], labels=batch['labels'], decoder_attention_mask=batch['decoder_attention_mask']).loss
                loss = loss.detach().cpu().numpy()
                
            
                outputs = model.generate(inputs=batch['input_ids'],
                                    attention_mask=batch['attention_mask'], max_length=self.args.generation_max_length, num_beams=self.args.generation_num_beams)
                predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
                # predictions = predictions.detach.cpu.numpy()


                self.all_losses.append(loss.item())
                self.all_predicitons.extend(predictions)
                del loss
                del predictions

        
        all_targets = [validation_dataset.dataset['output'][i][:] for i in range(len(validation_dataset.dataset[:]['output']))]
        metrics = self.compute_metrics(EvalPrediction(predictions=self.all_predicitons, label_ids=all_targets))
        # print(self.all_losses)
        metrics = denumpify_detensorize(metrics)
        metrics[f"{metric_key_prefix}_loss"] = statistics.mean(self.all_losses)

        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        return EvalLoopOutput(predictions=self.all_predicitons, label_ids=all_targets, metrics=metrics, num_samples=len(validation_dataset))


    def to_device(self, batch):
        for key_idx, key in enumerate(batch):        
            batch[key] = batch[key].to(self.args.device)

        return batch

            


def populate_config_file(json_config='training_args.json'):
    import dataclasses

    with open(json_config) as file: 
        config_dict = json.load(file)

    keys = {f.name for f in dataclasses.fields(Seq2SeqTrainingArguments)}
    values = {k: v for k, v in config_dict.items() if k in keys}


    return Seq2SeqTrainingArguments(**values)

         





