from inputEncoder import InputEncoder
from torch.utils.data.sampler import SequentialSampler
from torch.utils.data import dataloader, RandomSampler
import torch
import transformers
from trainingModel import TrainingModel
from preProcessor import PreProcessor
import sklearn
import os
os.add_dll_directory(
    "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.5/bin")


def evaluate(dataEncoded, trainingModel: TrainingModel, device):
    input_ids, predictions, true_labels, attentions = [], [], [], []
    trainingModel.model.eval()
    for i, batch_cpu in enumerate(dataEncoded):
        batch_gpu = (t.to(device) for t in batch_cpu)
        input_ids_gpu, attention_mask, labels = batch_gpu
        with torch.no_grad():
            loss, logits, hidden_states_output, attention_mask_output = trainingModel.model(
                input_ids=input_ids_gpu, attention_mask=attention_mask, labels=labels)
            logits = logits.cpu()
            prediction = torch.argmax(logits, dim=1).tolist()
            true_label = labels.cpu().tolist()
            input_ids_cpu = input_ids_gpu.cpu().tolist()
            # selection the last attention layer
            attention_last_layer = attention_mask_output[-1].cpu()
            # selection the last head attention of CLS token
            attention_softmax = attention_last_layer[:, -1, 0].tolist()
            input_ids += input_ids_cpu
            predictions += prediction
            true_labels += true_label
            attentions += attention_softmax
    return input_ids, predictions, true_labels, attentions


def train_model(dataEncoded, optimizer, scheduler, device, trainingModel: TrainingModel, epochs: int):

    trainingModel.model.train()
    for e in range(epochs):
        for i, batch in enumerate(dataEncoded):
            input_ids, attention_mask, labels = (
                b.type(torch.LongTensor).to(device) for b in batch)
            outputs = trainingModel.model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            # loss, logits, hidden_states_output, attention_mask_output = outputs
            loss = outputs[0]
            if i % 100 == 0:
                print("loss - {0}, iteration - {1}/{2}".format(loss, e + 1, i))
            trainingModel.model.zero_grad()
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainingModel.model.parameters(),
                                           parameters['max_grad_norm'])
            optimizer.step()
            scheduler.step()


epochs = 5
batch_size = 4
parameters = {
    'learning_rate': 2e-5,
    'num_warmup_steps': 1000,
    'num_training_steps': batch_size * epochs,
    'max_grad_norm': 1
}
torch.cuda.empty_cache()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
preProcessor = PreProcessor()
trainingModel = TrainingModel(preProcessor.n_of_labels())
trainingModel.model.to(device)
optimizer = transformers.AdamW(trainingModel.model.parameters(),
                               lr=parameters['learning_rate'], correct_bias=False)
scheduler = transformers.get_linear_schedule_with_warmup(optimizer,
                                                         num_warmup_steps=parameters['num_warmup_steps'],
                                                         num_training_steps=parameters['num_training_steps'])
x_train, y_train, x_test, y_test, x_val, y_val = preProcessor.fetch()
trainInputEncoded = InputEncoder(x_train, y_train, RandomSampler, batch_size)
valInputEncoded = InputEncoder(x_val, y_val, SequentialSampler, batch_size)
testInputEncoded = InputEncoder(x_test, y_test, None, batch_size)

train_model(trainInputEncoded.dataloaded, optimizer, scheduler,
            device, trainingModel, epochs)

input_ids, predictions, true_labels, attentions = evaluate(
    valInputEncoded.dataloaded, trainingModel.model, device)

print(sklearn.metrics.classification_report(true_labels, predictions))
