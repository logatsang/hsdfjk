import torch

from sentencepiece import SentencePieceProcessor
from typing import IO, Optional, Iterable

# en <-> fr
# en <-> de
class EuroparlDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        src_file: IO,
        src_tokenizer: SentencePieceProcessor,
        tgt_file: Optional[IO] = None,
        tgt_tokenizer: Optional[SentencePieceProcessor] = None
    ):

        print("Init dataset")

        self.src_tokenizer = src_tokenizer
        src_data = tokenize_lines(src_file.read().splitlines(), self.src_tokenizer)
        
        if tgt_file and tgt_tokenizer:
            self.tgt_tokenizer = tgt_tokenizer
            tgt_data = tokenize_lines(tgt_file.read().splitlines(), self.tgt_tokenizer)
        else:
            self.tgt_tokenizer = src_tokenizer
            tgt_data = src_data

        self.pairs = tuple(zip(src_data, tgt_data))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]
    
    def encode_src(self, text: str) -> torch.Tensor:
        return torch.tensor(self.src_tokenizer.encode(text))
    
    def decode_tgt(self, tokens: torch.Tensor) -> str:
        return self.tgt_tokenizer.decode(tokens.tolist())


def tokenize_lines(lines: Iterable[str], tokenizer: SentencePieceProcessor):
    return list(
        torch.tensor(tokenizer.encode(line.strip()), dtype=torch.long)
        for line in lines
    )


if __name__ == "__main__":
    with open('dataset/sample.txt', 'r', encoding='utf-8') as f:
        data = EuroparlDataset(f)

    print(data)
    print(len(data))
    print(data[2])