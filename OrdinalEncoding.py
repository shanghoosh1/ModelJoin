import numpy as np


class ordinalEncoding:
    def __init__(self):
        self.inverse_dict = {}
        self.main_dict = {}

    def fit(self, data):
        self.inverse_dict = {}
        self.main_dict = {}
        i = 0
        for d in data:
            d=tuple(d)
            if not d in self.main_dict:
                self.main_dict.update({d: i})
                self.inverse_dict.update({i: d})
                i += 1

    def transform(self, data):
        out = []
        for d in data:
            # d=tuple(d)
            if d[0] in self.main_dict:
                out.append(self.main_dict[d[0]])
            else:
                out.append(np.nan)
        return np.array(out)

    def fit_transform(self, data):
        self.inverse_dict = {}
        self.main_dict = {}
        out=[]
        i = 0
        for d in data:
            d=tuple(d)
            if not d in self.main_dict:
                self.main_dict.update({d: i})
                self.inverse_dict.update({i: d})
                out.append(i)
                i += 1
            else:
                out.append(self.main_dict[d])
        return np.array(out)
    def fit_transform_largeData(self, data):
        self.inverse_dict = {}
        self.main_dict = {}
        out=[]
        i = 0
        for d in data:
            if not d[0] in self.main_dict:
                self.main_dict.update({d[0]: i})
                self.inverse_dict.update({i: d[0]})
                out.append(i)
                i += 1
            else:
                out.append(self.main_dict[d[0]])
        return np.array(out)
    def fit_update_transform(self, data):
        out=[]
        i = len(self.main_dict)
        for d in data:
            d=d[0]
            if not d in self.main_dict:
                self.main_dict.update({d: i})
                self.inverse_dict.update({i: d})
                out.append(i)
                i += 1
            else:
                out.append(self.main_dict[d])
        return np.array(out)

    def inverse_transform(self, data):
        data=np.array(data).reshape(len(data),)
        out=[]
        for i in data:
            if i in self.inverse_dict:
                out.append(self.inverse_dict[i])
            else:
                out.append(np.nan)
        return np.array(out)