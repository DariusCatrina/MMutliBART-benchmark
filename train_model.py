from trainer import MMultiBART_Trainer, populate_config_file
from torch.utils.data import DataLoader
from Data.Summarization import Summarization_Wrapper
from MMultiBART.multibart import MultiBART as MMultiBART
#from MultiBART.MultiBARTEncoder import MultiBARTEncoder

from transformers import BartForConditionalGeneration, BartTokenizerFast

import numpy as np

from utils import compute_rouge

import wandb
import argparse

training_dataset=0
validation_dataset=0
tokenizer = BartTokenizerFast.from_pretrained("facebook/bart-base")

def load_datasets(bart_seq_len, seq_len, batch_size, decoder_length, num_decoders, dataset_name="gov_report"):
    global training_dataset, validation_dataset
    training_dataset = Summarization_Wrapper(dataset_name = dataset_name, 
                                             split='train',
                                             bart_seq_len=bart_seq_len,
                                             seq_len=seq_len,
                                             fast_tokenizer=True,
                                             tensor_type='pt',
                                             num_decoders=num_decoders)
    training_dataset.target_len = decoder_length
    validation_dataset = Summarization_Wrapper(dataset_name = dataset_name, 
                                             split='validation',
                                             bart_seq_len=bart_seq_len,
                                             seq_len=seq_len,
                                             fast_tokenizer=True,
                                             tensor_type='pt',
                                             num_decoders=num_decoders)

    validation_dataset.target_len = decoder_length
    return (training_dataset, validation_dataset)



def load_dataloaders(bart_seq_len, seq_len, batch_size, decoder_length, num_decoders, dataset_name="gov_report"):

    training_dataset, validation_dataset = load_datasets(bart_seq_len, seq_len, batch_size, dataset_name="gov_report", decoder_length=decoder_length, num_decoders=num_decoders)

    training_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)

    return (training_dataloader, validation_dataloader)



def compute_metrics(eval_preds):
    preds, labels = eval_preds
    #compute rouge
    scores = compute_rouge(preds, labels)


    scores["rouge_mean"] = (scores["rouge1"] * scores["rouge2"] * scores["rougeL"]) ** (1.0 / 3.0)
       
    return scores


def main(seq_len, batch_size, number_of_decoders, decoder_length, bart_seq_len, wandb_proj, wandb_ent):
    gpu = '|A6000'
    name = 'MMultiBART|E:'+str(bart_seq_len)+ '/'+str(seq_len)+'|D:'+str(decoder_length)+'/'+str(number_of_decoders)+'|b'+str(batch_size)+gpu
    print(name)
    print('WANDB INIT')
    wandb.init(project=wandb_proj, entity=wandb_ent, name=name)


    print('MODEL INIT')
    PRETRAINED_BART = BartForConditionalGeneration.from_pretrained("facebook/bart-base")   
    model = MMultiBART(encoder_len=bart_seq_len, decoder_len=decoder_length, num_decoders=number_of_decoders, tokenizer=tokenizer)
    
    model.apply_pretrained_weights(PRETRAINED_BART)
    # MultiBARTEncoder.bart_seq_len = bart_seq_len
    

    print('DATA INIT')
    #Set the training/eval dataloaders for the trainer class
    training_dataloader, validation_dataloader = load_dataloaders(bart_seq_len=bart_seq_len,batch_size=batch_size, seq_len=seq_len, decoder_length=decoder_length, num_decoders=number_of_decoders)
    MMultiBART_Trainer.train_dataloader = training_dataloader
    MMultiBART_Trainer.validation_dataloader = validation_dataloader
    MMultiBART_Trainer.number_of_decoders = number_of_decoders

    print('TRAINING ARGS INIT')
    seq2seq_config = populate_config_file()
    print('TRAINING INIT')
    trainer = MMultiBART_Trainer(
        model = model,
        args = seq2seq_config,
        tokenizer = tokenizer,
        compute_metrics = compute_metrics
    )
    
    if seq2seq_config.do_train == True:
        train_result = trainer.train(ignore_keys_for_eval=["past_key_values", "encoder_last_hidden_state"])
        trainer.save_model()

        metrics = train_result.metrics

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    if seq2seq_config.do_eval == True:
        metrics = trainer.evaluate(max_length=seq2seq_config.generation_max_length, num_beams=seq2seq_config.generation_num_beams, metric_key_prefix="eval", ignore_keys=["past_key_values", "encoder_last_hidden_state"])

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)


if __name__ == '__main__':


    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str,
        help='Name of dataset from HuggingFace or the local directory path to the datasets')
    parser.add_argument("--seq_len", type=int,
        help='Input sequence lengths')
    parser.add_argument("--bsz", type=int,
        help='Batch sizes')
    parser.add_argument("--number_of_decoders", type=int,
        help='Number of Decoders')
    parser.add_argument("--bart_seq_len", type=int,
        help='Bart sequence length(i.e. 256, 512, 1024)')
    parser.add_argument("--decoder_length", type=int,
        help='Length fo the target per decoder(i.e. 128, 512, etc.)')
    parser.add_argument("--wandb_proj", type=str, default='MultiBART(MEMD)',
        help='Optional argument for name of the project on wandb')
    parser.add_argument("--wandb_ent", type=str, default='darius-catrina',
        help='Optional argument for username or entity on wandb')

    args = parser.parse_args()

    main(seq_len=args.seq_len, batch_size=args.bsz, number_of_decoders=args.number_of_decoders, bart_seq_len=args.bart_seq_len, wandb_proj=args.wandb_proj, wandb_ent=args.wandb_ent, decoder_length=args.decoder_length)



