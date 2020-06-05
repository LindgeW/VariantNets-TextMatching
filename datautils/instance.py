
class Instance(object):
    def __init__(self, sent1, sent2, label):
        self.sent1 = sent1
        self.sent2 = sent2
        self.label = label

    def __repr__(self):
        return f'sent1:{self.sent1}, sent2:{self.sent2}, label:{self.label}'

    def __str__(self):
        return f'sent1:{self.sent1}, sent2:{self.sent2}, label:{self.label}'
