from operation import *
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from pydantic import BaseModel
from typing import Literal, Optional
import torch

class ClassifyTextArgs(BaseModel):
    model: str
    text_column: str
    labels: list[str]
    batch_size: Optional[int] = 256
    max_sequence_length: Optional[int] = 0

class ClassifyTextOperation(Operation):
    def __call__(self, dataset: Dataset | DatasetDict, **kwargs) -> Dataset | DatasetDict:
        self.args = ClassifyTextArgs(**kwargs)
        self.trace(f"Loading classifier model from {self.args.model}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(self.args.model, torch_dtype=torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(self.args.model)

        self.device = "cuda"
        self.model.to(self.device)

        if self.args.max_sequence_length == 0:
            self.args.max_sequence_length = self.model.config.max_position_embeddings
        
        with torch.no_grad():
            dataset = dataset.map(lambda x: self.batched_classify(x), 
                                  load_from_cache_file=False, 
                                  batched=True,
                                  batch_size=self.args.batch_size,
                                  desc="Classifying " + self.args.text_column)
        return dataset
    
    def batched_classify(self, examples):
        tokens = self.tokenizer(examples[self.args.text_column], 
                                    padding="max_length", 
                                    max_length=self.args.max_sequence_length, 
                                    truncation=True,
                                    return_tensors="pt").to(self.device)
        output = self.model(**tokens).logits.to("cpu")
        if output.shape[1] > 1:
            output = output.softmax(1)
        output = output.transpose(0, 1)
        return { label: output[i] for i, label in enumerate(self.args.labels) }

ClassifyTextOperation.register("classify_text")