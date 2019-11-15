import numpy as np


class Dict(object):
    def __init__(self, data=None, lower=False):

        self.idxToLabel = {}
        self.labelToIdx = {}
        self.frequencies = {}
        self.lower = lower
        self.special = []
        if data is not None:
            if type(data) == str:
                self.loadFile(data)
            else:
                self.addSpecials(data)

    def size(self):
        return len(self.idxToLabel)

    def get_num_from_end(self, sent):
        word = ""
        _i = len(sent) - 1
        while _i >= 0:
            if sent[_i].isdigit():
                word = sent[_i] + word
            else:
                return int(word), _i
            _i -= 1

    def loadFile(self, filename, partion_line_num=None):
        with open(filename, "r", encoding="utf-8") as fd:

            fd_lines = fd.readlines()
            if partion_line_num is not None:
                len_line = partion_line_num
            else:
                len_line = len(fd_lines)
            for _index_fd_lines in range(len_line):
                fields = fd_lines[_index_fd_lines].strip("\n")

                idx, space_index = self.get_num_from_end(fields)
                label = fields[:space_index]

                self.add(label, idx)

    def writeFile(self, filename):
        with open(filename, "w", encoding="utf-8") as file:
            for i in range(self.size()):
                label = self.idxToLabel[i]
                if label[-1] == "\n":
                    print("!!!enter-after-label:\n ", label)
                file.write('%s %d\n' % (label, i))

    def lookup(self, key, default=None):
        key = key.lower() if self.lower else key
        try:
            return self.labelToIdx[key]
        except KeyError:
            return default

    def getLabel(self, idx, default=None):
        try:
            return self.idxToLabel[idx]
        except KeyError:
            return default

    def addSpecial(self, label, idx=None):
        idx = self.add(label, idx)
        self.special += [idx]

    def addSpecials(self, labels):

        for label in labels:
            self.addSpecial(label)

    def add(self, label, idx=None):
        label = label.lower() if self.lower else label
        if idx is not None:
            self.idxToLabel[idx] = label
            self.labelToIdx[label] = idx
        else:
            if label in self.labelToIdx:
                idx = self.labelToIdx[label]
            else:
                idx = len(self.idxToLabel)
                self.idxToLabel[idx] = label
                self.labelToIdx[label] = idx

        if idx not in self.frequencies:
            self.frequencies[idx] = 1
        else:
            self.frequencies[idx] += 1

        return idx

    def prune(self, size):

        if size >= self.size():
            return self

        freq = [self.frequencies[i] for i in range(len(self.frequencies))]
        idx = np.argsort(freq)[::-1].tolist()
        newDict = Dict()
        newDict.lower = self.lower

        for i in self.special:
            newDict.addSpecial(self.idxToLabel[i])

        for i in idx[:size]:
            newDict.add(self.idxToLabel[i])

        return newDict

    def convertToIdx(self, labels, unkWord, bosWord=None, eosWord=None):

        vec = []

        if bosWord is not None:
            vec += [self.lookup(bosWord)]

        unk = self.lookup(unkWord)

        if self.lower == True:
            labels = [label.lower() for label in labels]

        vec += [self.lookup(label, default=unk) for label in labels]

        if eosWord is not None:
            vec += [self.lookup(eosWord)]

        return vec

    def convertToLabels(self, idx, stop):

        labels = []

        for i in idx:
            labels += [self.getLabel(i)]
            if i == stop:
                break

        return labels
