from operation import *
from transformers import AutoTokenizer, AutoConfig
from pydantic import BaseModel
from typing import Literal, Optional

type TokenizeSplitType = Literal["paragraph", "sentence"]

class TokenizeTextArgs(BaseModel):
    model: str
    text_column: str
    split_type: Optional[TokenizeSplitType] = "paragraph"
    max_sequence_length: Optional[int] = 0

class TokenizeTextOperation(Operation):
    def __call__(self, dataset: Dataset | DatasetDict, **kwargs) -> Dataset | DatasetDict:
        self.args = TokenizeTextArgs(**kwargs)
        self.trace(f"Loading tokenizer from {self.args.model}...")
        self.config = AutoConfig.from_pretrained(self.args.model)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.total_tokens = 0

        if self.args.max_sequence_length == 0:
            self.args.max_sequence_length = self.config.max_position_embeddings
        dataset = dataset.map(lambda x: self.batched_tokenize(x), 
                              batched=True,
                              remove_columns=self.args.text_column, 
                              desc="Tokenizing")
        self.trace(f"Encoded {self.total_tokens} tokens")
        return dataset

    def batched_tokenize(self, examples):
        tokenized = { "input_ids": [], "attention_mask": [] }
        for batch in examples[self.args.text_column]:
            tokens = self.tokenizer(batch, max_length=self.args.max_sequence_length, truncation=True)
            tokenized["input_ids"].append(tokens["input_ids"])
            tokenized["attention_mask"].append(tokens["attention_mask"])
            self.total_tokens += self.args.max_sequence_length
        return tokenized

TokenizeTextOperation.register("tokenize_text")