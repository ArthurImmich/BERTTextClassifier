from transformers import BertForSequenceClassification


class TrainingModel:

    def __init__(self, n_of_labels):
        self.model = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                                   num_labels=n_of_labels,
                                                                   output_attentions=True,
                                                                   output_hidden_states=True,
                                                                   )
