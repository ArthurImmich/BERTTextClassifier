import os

os.add_dll_directory(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")
from transformers import (AdamW, BertForSequenceClassification, BertTokenizer,
                          DataCollatorWithPadding, Trainer, TrainingArguments,
                          get_linear_schedule_with_warmup)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from sklearn.datasets import fetch_20newsgroups
from numpy.lib.function_base import average
from datasets import load_metric
import torch
import numpy as np


torch.cuda.empty_cache()

####################################
# Parameters
####################################

epochs = 15
batch_size = 32
max_length = 200
parameters = {
    'learning_rate': 2e-5,
    'num_warmup_steps': 200,
    'num_training_steps': batch_size * epochs,
    'max_grad_norm': 1
}

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

####################################
# Data Preparation
####################################


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels=None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach()
                for key, val in self.encodings.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])


def fetchEquals(x, y, n, s):
    new_x = []
    new_y = []
    new_x_helper = []
    new_y_helper = []
    for j in range(n):
        new_x_helper.clear()
        new_y_helper.clear()
        for i in range(len(x)):
            if y[i] == j:
                new_x_helper.append(x[i])
                new_y_helper.append(y[i])
        new_x.extend(new_x_helper[:s])
        new_y.extend(new_y_helper[:s])
    return new_x, new_y


tokenizer = BertTokenizer.from_pretrained(
    "bert-base-uncased", do_lower_case=True)


def tokenizer_function(e):
    return tokenizer(e, truncation=True, padding=True, max_length=max_length, return_tensors='pt')


newsgroups = fetch_20newsgroups(
    subset='all', remove=('headers', 'footers', 'quotes'))
texts, labels = fetchEquals(
    newsgroups.data, newsgroups.target, len(newsgroups.target_names), 100)
x_train, x_test, y_train, y_test = train_test_split(
    texts, labels, train_size=1400, test_size=600, shuffle=True, stratify=labels)
x_train, x_val, y_train, y_val = train_test_split(
    x_train, y_train, test_size=0.1, shuffle=True, stratify=y_train)

print(f"train: text - {len(x_train)} targets - {len(y_train)}")
print(f"test: text - {len(x_test)} targets - {len(y_test)}")
print(f"val: text - {len(x_val)} targets - {len(y_val)}")

x_train_tokenized = tokenizer_function(x_train)
x_val_tokenized = tokenizer_function(x_val)
x_test_tokenized = tokenizer_function(x_test)
train_dataset = Dataset(x_train_tokenized, y_train)
val_dataset = Dataset(x_val_tokenized, y_val)
test_dataset = Dataset(x_test_tokenized, y_test)

####################################
# Model
####################################

model = BertForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(newsgroups.target_names)).to(device)
optimizer = AdamW(model.parameters(),
                  lr=parameters['learning_rate'], correct_bias=True)
scheduler = get_linear_schedule_with_warmup(
    optimizer, num_warmup_steps=parameters['num_warmup_steps'], num_training_steps=parameters['num_training_steps'])
data_collator = DataCollatorWithPadding(tokenizer)

####################################
# Compute Metrics
####################################
# Here we simply take the model’s output,
# find the maximum value, and compute the metrics with respect to the corresponding label.


def compute_metrics(p):
    preds = np.argmax(p.predictions, axis=-1)
    accuracy = accuracy_score(y_pred=preds, y_true=p.label_ids)
    # recall = recall_score(y_pred=preds, y_true=p.label_ids, average="weighted"),
    precision = precision_score(
        y_pred=preds, y_true=p.label_ids, average="weighted")
    f1 = f1_score(y_pred=preds, y_true=p.label_ids, average="weighted")
    return {"accuracy": accuracy, "precision": precision, "f1": f1}


####################################
# MAIN
####################################
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

train_args = TrainingArguments(
    f"test-squad",
    num_train_epochs=epochs,
    learning_rate=parameters['learning_rate'],
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_dir='logs',
)

trainer = Trainer(
    model=model,
    args=train_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    optimizers=(optimizer, scheduler),
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
output = trainer.predict(test_dataset)
preds = np.argmax(output.predictions, axis=-1)
print(f"Test acurracy: {accuracy_score(y_pred=preds, y_true=y_test)}")
