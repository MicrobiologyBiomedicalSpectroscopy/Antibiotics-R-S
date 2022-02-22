from multiprocessing.pool import Pool

import matplotlib.pyplot as plt
import seaborn

from postprocess import *


class MultiClassPostProcess:
    threshold = 0

    def __init__(self, scors_list=None):
        pool = Pool()
        voted_data = pool.map(_LLr_voting, scors_list)
        pool.close()
        pool.join()
        self.voted_data = voted_data.pop(0)
        for i in voted_data:
            self.voted_data = pd.concat([self.voted_data, i], axis=1)

    def heat_map_plot(self):
        seaborn.heatmap((self.voted_data.corr()))
        print(self.voted_data.corr())
        plt.show()

    def voting_differences(self):
        seaborn.heatmap((self.voted_data))
        plt.show()


    def __add__(self, other):
        if not isinstance(other, list):
            other = [other]
        pool = Pool()
        other = pool.map(_LLr_voting, other)
        pool.close()
        pool.join()
        for i in other:
            self.voted_data = pd.concat([self.voted_data, i], axis=1, copy=True)


def _LLr_voting(scorse):
    model = PostProcess(scorse)
    voted_data = model.LLR(MultiClassPostProcess.threshold)
    return voted_data["labels"] == voted_data["voting"]
