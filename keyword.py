from keybert import KeyBERT
import pandas as pd
import matplotlib.pyplot as plt

class keywords():
    def __init__(self,name_file):
        self.tweets = pd.read_excel(f'sheets/{name_file}.xlsx')['Tweet']
        self.model = KeyBERT()
        self.keywords = self.__getkeywords__()

    def __getkeywords__(self):
        keywords_set = self.model.extract_keywords(self.tweets, keyphrase_ngram_range=(1, 1),top_n = 2)
        keywords = []
        for tweet in keywords_set:
            for keyword in tweet:
                keywords.append(keyword[0])
        return pd.Series(keywords).value_counts(ascending = True)[:10]

    def plot(self):
        fig, ax = plt.subplots(figsize =(16, 20))
        ax.barh(self.keywords.keys(), self.keywords)
        plt.show()
