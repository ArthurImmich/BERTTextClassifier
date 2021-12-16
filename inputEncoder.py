import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import BertTokenizer

# Encoding input


class InputEncoder():

    def __init__(self, X, Y, Sampler, batch_size):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
        self.X = X
        self.Y = torch.tensor(Y)
        X_features = self.tokenizer.batch_encode_plus(self.X,
                                                      # Add [CLS] and [SEP] tokens
                                                      add_special_tokens=True,
                                                      # Add [PAD]s
                                                      padding=True,
                                                      max_length=512,
                                                      return_token_type_ids=False,
                                                      return_attention_mask=True,  # Generate the attention mask
                                                      truncation=True,  # Truncate data beyond max length
                                                      return_tensors='pt',  # Retuns PyTorch tensor format
                                                      )
        dataset = TensorDataset(
            X_features['input_ids'], X_features['attention_mask'], self.Y)

        if Sampler != None:
            self.dataloaded = DataLoader(dataset,
                                         sampler=Sampler(dataset), batch_size=batch_size)
