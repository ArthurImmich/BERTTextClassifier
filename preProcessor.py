from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split

# loading and prepocessing the data


class PreProcessor():

    def __init__(self):
        self.newsgroups = fetch_20newsgroups(
            subset='all', remove=('headers', 'footers', 'quotes'))
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.newsgroups.data, self.newsgroups.target, test_size=0.3, shuffle=True)
        self.x_train, self.x_validation, self.y_train, self.y_validation = train_test_split(
            self.x_train, self.y_train, test_size=0.2, shuffle=True)

    def fetch(self):
        return self.x_train, self.y_train, self.x_test, self.y_test, self.x_validation, self.y_validation

    def n_of_labels(self):
        return len(self.newsgroups.target_names)
