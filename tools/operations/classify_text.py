from operation import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer, AutoConfig
from pydantic import BaseModel
from typing import Literal, Optional
import torch

class ClassifyTextArgs(BaseModel):
    model: str
    text_column: str
    labels: list[str] | dict[str, str]
    max_sequence_length: Optional[int] = 0

class ClassifyTextOperation(Operation):
    def __call__(self, dataset: Dataset | DatasetDict, **kwargs) -> Dataset | DatasetDict:
        self.args = ClassifyTextArgs(**kwargs)
        self.trace(f"Loading classifier model from {self.args.model}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)
        self.config = AutoConfig.from_pretrained(self.args.model)

        self.device = "cuda"
        self.model.to(self.device)

        if self.args.max_sequence_length == 0:
            self.args.max_sequence_length = self.config.max_position_embeddings
        
        with torch.no_grad():
            dataset = dataset.map(lambda x: self.batched_classify(x), batched=True, batch_size=128)
        return dataset
    
    def batched_classify(self, examples):
        for batch in examples[self.args.text_column]:
            tokens = self.tokenizer(batch, 
                                    padding="max_length", 
                                    max_length=self.args.max_sequence_length, 
                                    truncation=True, 
                                    return_tensors="pt").to(self.device)
            output = self.model(**tokens)
            # TODO: Actually do something with the labels
        return { }

ClassifyTextOperation.register("classify_text")