from transformers import BertTokenizer
from preProcessor import PreProcessor

# Encoding input


class InputEncoder():

    def __init__(self, X_test):
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-uncased", do_lower_case=True)
        self.X_test_raw_data = X_test
        self.X_test_features = list()

    def getFeatures(self):
        for x_test_raw in self.X_test_raw_data:
            self.X_test_features.append(self.tokenizer.encode_plus(text=x_test_raw, add_special_tokens=True,  # Add [CLS] and [SEP] tokens
                                                                   # Add [PAD]s
                                                                   padding=True,
                                                                   max_length=512,
                                                                   return_token_type_ids=False,
                                                                   return_attention_mask=True,  # Generate the attention mask
                                                                   truncation=True,  # Truncate data beyond max length
                                                                   return_tensors='pt',  # Retuns PyTorch tensor format
                                                                   ))
            self.X_test_features[-1]['attention_mask'] = self.X_test_features[-1]["attention_mask"].flatten()
            self.X_test_features[-1]['input_ids'] = self.X_test_features[-1]["input_ids"].flatten()


news = PreProcessor()
x_train, x_test, y_train, y_test, x_validation, y_validation = news.fetch()
InputEncoder(x_train).getFeatures()
