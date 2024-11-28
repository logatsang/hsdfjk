import torch

import time

import sentencepiece as spm
from tqdm import tqdm

from t5_pytorch import T5
from dataset import EuroparlDataset
from torch.utils.data import DataLoader

from typing import List

from pytorch_beam_search import seq2seq


device = torch.device('cpu') # torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_input_target_split(target_tokens):
    inputs = target_tokens[:,:-1]
    targets = target_tokens[:,1:].contiguous().flatten()
    return inputs, targets


def schedule_rate(step, model_size, factor, warmup):
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )



def rnn_collate(seq):
    source_x, target_y = zip(*seq)
    source_lens = torch.tensor([len(f) for f in source_x])
    source_x = torch.nn.utils.rnn.pad_sequence(
        source_x, padding_value=3 #pad
    )
    target_y = torch.nn.utils.rnn.pad_sequence(
        target_y, padding_value=3 #pad
    )
    return source_x, source_lens, target_y


class Runner:
    def __init__(self, dataset: EuroparlDataset):
        self.model = T5(
            dim = 768,
            #max_seq_len = 1024,
            enc_num_tokens = 32000,
            enc_depth = 6,
            enc_heads = 12,
            enc_dim_head = 64,
            enc_mlp_mult = 4,
            dec_num_tokens = 32000,
            dec_depth = 6,
            dec_heads = 12,
            dec_dim_head = 64,
            dec_mlp_mult = 4,
            dropout = 0.,
            tie_token_emb = True
        )
        print("Init")
        self.dataloader = DataLoader(
            dataset, batch_size=5,
            collate_fn=rnn_collate
        )


    def train_for_epoch(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LambdaLR,
        criterion: torch.nn.CrossEntropyLoss,
        accum_iter: int = 20
    ):
        
        iterations = 0
        total_loss = 0

        for source, source_lens, target in tqdm(self.dataloader):
            iterations += 1

            source = source.to(device)
            source_lens = source_lens.to(device)
            target = target.to(device)

            model_inputs, criterion_targets = train_input_target_split(target)

            logits = self.model(source, model_inputs).flatten(0, 1)
            loss = criterion(logits, criterion_targets)

            total_loss += loss
            loss = loss / accum_iter
            loss.backward()

            if iterations % accum_iter == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
        
        return total_loss/iterations, iterations


    def train(
        self
    ):
        print("Starting training")
        optimizer = torch.optim.AdamW(
            self.model.parameters()
        )

        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer=optimizer,
            lr_lambda=lambda step: schedule_rate(
                step, 512, 1.0, 300
            )
        )

        criterion = torch.nn.CrossEntropyLoss(
            label_smoothing=0.1,
            ignore_index=3, # padding index
            reduction='mean'
        )

        self.model.to(device)
        criterion.to(device)

        self.model.train()

        cur_step = 0
        cur_epoch = 1

        start = time.time()
        while cur_epoch <= 50:
            # self.model.train()
            print(f"Epoch {cur_epoch} training")

            loss, num_steps = self.train_for_epoch(
                # self.dataloader,
                optimizer,
                scheduler,
                criterion
            )
            cur_step += num_steps

            torch.cuda.empty_cache()

            t = time.strftime("%H:%M:%S", time.gmtime(time.time() - start))

            print(f"Epoch {cur_epoch}: loss={loss}, time={t}")

            # self.model.eval()
            torch.save(self.model.state_dict, f'models/epoch_{cur_epoch}')

            cur_epoch += 1

        print(f"Finished {cur_epoch-1} epochs")
    

    def translate(self, text) -> List[str]:
        self.model.eval()
        source_tokens = self.dataloader.dataset.encode_src(text)

        predictions, log_probabilities = seq2seq.beam_search(self.model, source_tokens) 
        predictions = [
            self.dataloader.dataset.decode_tgt(
                prediction[:prediction.index(3)]
            )
            for prediction in predictions[0].tolist()
        ]
        
        return predictions


    

if __name__ == "__main__":
    src_sp = spm.SentencePieceProcessor(model_file='spm_en.model')
    tgt_sp = src_sp

    with open('dataset/en_all.en', 'r', encoding='utf-8') as f:
        dataset = EuroparlDataset(f, src_tokenizer=src_sp)

    runner = Runner(dataset)

    runner.train()
