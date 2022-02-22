import math
import warnings
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")


class PostProsses:
    """
     post prossesin tool
    """

    def __init__(self, df, zeros="0", ones="1"):
        """
        post possessing decision logic tool
        :param df:  pandas data frame of probabilities labels and groups
        :param zeros: the label of the zeros label in you're log
        :param ones: the label of the ones label in you're log
        """
        self._zeros = zeros
        self._ones = ones
        self.df = df.copy()
        self.groups_ = df["group"].values
        del df["group"]
        self.labels_ = df["labels"].values
        del df["labels"]
        self.data_ = df.values
        self.voted = None
        self.fpt_ = None
        self.tpr_ = None
        self.threshold_ = None
        self.auc_ = None

    def LLR_threshold(self, show_roc=False):

        """
        shows the ROC curve for LLR
        :return: the voting between groups inside sigmoid (type: df)
        """
        df = self.df.copy()
        groups = df.loc[:, "group"].values
        groups = np.unique(groups)

        voting = []
        labels = []

        for group in groups:
            tamp = df.loc[df.loc[:, "group"] == group, :]
            normalize_size = tamp.shape[0]
            labels.append(tamp.iloc[0, 2].copy())
            tamp = tamp.loc[:, self._zeros:self._ones].values
            desigen_logic = np.log2(tamp)
            LLR_zero = np.sum(desigen_logic[:, 0])/normalize_size
            LLR_ones = np.sum(desigen_logic[:, 1])/normalize_size
            LLR = LLR_ones - LLR_zero
            LLR = _sigmoid(LLR)
            voting.append(LLR)

        df_save = pd.DataFrame()
        df_save["group"] = groups
        df_save["labels"] = labels
        df_save["voting"] = voting
        labels = np.array(labels)
        voting = np.array(voting)
        self.fpt_, self.tpr_, self.threshold_ = metrics.roc_curve(labels, voting)
        self.auc_ = metrics.auc(self.fpt_, self.tpr_)
        if show_roc:
            self.show_roc()
        return df_save

    def show_roc(self, name=None, legend_location=4, fig_size_x=7, fig_size_y=7):
        """
        show the roc of your classification
        :param name:the name that you want to give to you're pic
        :param legend_location: which corner you put the legend
        :param fig_size_x: the white of the figure
        :param fig_size_y:the height of the figure
        :return: None
        """

        plt.figure(figsize=(fig_size_x, fig_size_y))
        plt.rc("font", family="Times New Roman", size=16)
        plt.rc('axes', linewidth=2)
        plt.plot(self.fpt_, self.tpr_, label="(AUC = %0.2f)" % self.auc_)
        plt.plot([0, 1], [0, 1], "--r")
        plt.xlabel("1-Specificity", fontdict={"size": 21})
        plt.ylabel("Sensitivity", fontdict={"size": 21})
        plt.legend(loc=legend_location)
        if name is None:
            plt.show()
        else:
            plt.savefig(str(name + ".png"))

    def LLR(self, threshold=0):
        """
        classing by log likelihood ratio vote
        :param threshold: the threshold for disease who is one (type: float)
        :return: the voting between groups (type: df)
        """
        df = self.df.copy()
        groups = df.loc[:, "group"].values
        groups = np.unique(groups)

        voting = []
        labels = []

        for group in groups:
            tamp = df.loc[df.loc[:, "group"] == group, :]
            normalize_size=tamp.shape[0]
            labels.append(tamp.iloc[0, 2].copy())
            tamp = tamp.loc[:, self._zeros:self._ones].values
            desigen_logic = np.log2(tamp)
            LLR_zero = np.sum(desigen_logic[:, 0])/normalize_size
            LLR_ones = np.sum(desigen_logic[:, 1])/normalize_size
            LLR = LLR_ones - LLR_zero
            LLR = _sigmoid(LLR)
            if LLR <= threshold:
                voting.append(0)
            else:
                voting.append(1)
        df_save = pd.DataFrame()
        df_save["group"] = groups
        df_save["labels"] = labels
        df_save["voting"] = voting
        labels = np.array(labels)
        voting = np.array(voting)
        self.fpt_, self.tpr_, self.threshold_ = metrics.roc_curve(labels, voting)
        self.auc_ = metrics.auc(self.fpt_, self.tpr_)
        # self._classification_local_report(labels, voting)
        return df_save

    def majority_vote(self, threshold=0.5):
        """
        classing by majority vote
        :param threshold: the threshold for disease who is one (type: float between 0 to 1)
        :return: the voting between groups (type: df)
        """
        df = self.df.copy()
        groups = df.loc[:, "group"].values
        groups = np.unique(groups)

        voting = []
        labels = []

        for group in groups:
            tamp = df.loc[df.loc[:, "group"] == group, :]
            labels.append(tamp.iloc[0, 2].copy())
            tamp = tamp.loc[:, self._zeros:self._ones].values
            tamp = tamp > threshold
            tamp = tamp + np.zeros([tamp.shape[0], tamp.shape[1]])
            onse = np.sum(tamp[:, 1])
            zerose = np.sum(tamp[:, 0])
            if onse > zerose:
                voting.append(1)
            else:
                voting.append(0)

        df_save = pd.DataFrame()
        df_save["group"] = groups
        df_save["labels"] = labels
        df_save["voting"] = voting
        labels = np.array(labels)
        voting = np.array(voting)
        self.fpt_, self.tpr_, self.threshold_ = metrics.roc_curve(labels, voting)
        self.auc_ = metrics.auc(self.fpt_, self.tpr_)
        self._classification_local_report(labels, voting)
        return df_save

    def _classification_local_report(self, labels, predictions):
        """
        analise the classification that currently happened
        :param labels: the labels of ech group
        :param predictions: the prediction of ech group
        :return: None
        """
        self.confusion_matrix_ = metrics.confusion_matrix(labels, predictions)
        print(self.confusion_matrix_)
        self.classification_report_ = metrics.classification_report(labels, predictions)
        print(self.classification_report_)

    def t_test(self, removing_threshold=0.05):
        df = self.df[self._ones].values.copy()
        val, pdf_count = np.unique(df, return_counts=True)
        count = np.max(pdf_count) * removing_threshold
        count = np.where(pdf_count >= count)
        val = val[count]
        seen = []
        for i in val:
            tamp = np.where(df == i)[0].tolist()
            seen.extend(tamp)
        seen = np.array(seen)
        df = self.df.iloc[seen, :]
        self.__init__(df, self._zeros, self._ones)

    def optimal_cut_point_on_roc_(self, delta_max=0.8, plot_point_on_ROC=False):
        """
        print the optimal cut on you're roc curve
        :param delta_max: the maximum delta between tpr and fpr (type: flute between 0 to 1)
        :param plot_point_on_ROC: is you like to show the roc curve now (type:bool)
        :return: report on you're optimal working point (type: dictionary)
        """
        # tpr = self.fpt_
        # fpr = self.tpr_
        fpr = self.fpt_
        tpr = self.tpr_
        n_n = self.labels_[self.labels_ == 0].shape[0]
        n_p = self.labels_[self.labels_ == 1].shape[0]
        # sen = fpr[fpr > 0.55]
        # spe = 1 - tpr[fpr > 0.55]
        sen = tpr[fpr < 0.5]
        spe = 1 - fpr[fpr < 0.5]

        delt = abs(sen - spe)
        ix_1 = np.argwhere(delt <= delta_max)

        acc = (n_p / (n_p + n_n)) * sen[ix_1] + (n_n / (n_p + n_n)) * spe[ix_1]
        acc_max_index = ix_1[np.argmax(acc)]
        best_point = (1 - spe[acc_max_index], sen[acc_max_index])
        auc = np.around(np.trapz(self.tpr_, self.fpt_), 2)

        recall_1 = sen[acc_max_index]
        recall_2 = spe[acc_max_index]
        precision_1 = (n_p * sen[acc_max_index]) / (n_p * sen[acc_max_index] + n_n * (1 - spe[acc_max_index]))
        precision_2 = (n_n * spe[acc_max_index]) / (n_n * spe[acc_max_index] + n_p * (1 - sen[acc_max_index]))

        report = {"auc": np.around(auc, 2), "acc": np.around(acc.max(), 2), "recall_1": np.around(recall_1, 2),
                  "recall_2": np.around(recall_2, 2), "precision_1": np.around(precision_1, 2),
                  "precision_2": np.around(precision_2, 2)
                  }
        _plot_optimal_cut(self.fpt_, self.tpr_, best_point)
        # if plot_point_on_ROC:
        #     p = Process(target=ClassificationPreprocessing, args=(fpr, tpr, best_point,))
        #     p.start()

        return report

    def optimal_cut_point_on_roc_old_(self, delta_max=0.8, plot_point_on_ROC=False):
        """
        print the optimal cut on you're roc curve
        :param delta_max: the maximum delta between tpr and fpr (type: flute between 0 to 1)
        :param plot_point_on_ROC: is you like to show the roc curve now (type:bool)
        :return: report on you're optimal working point (type: dictionary)
        """
        tpr = self.fpt_
        fpr = self.tpr_
        # fpr = self.fpt_
        # tpr = self.tpr_
        n_p = self.target[self.target == 0].shape[0]
        n_n = self.target[self.target == 1].shape[0]
        sen = fpr[fpr > 0.55]
        spe = 1 - tpr[fpr > 0.55]
        # sen = tpr[fpr < 0.5]
        # spe = 1 - fpr[fpr < 0.5]

        delt = abs(sen - spe)
        ix_1 = np.argwhere(delt <= delta_max)

        acc = (n_p / (n_p + n_n)) * sen[ix_1] + (n_n / (n_p + n_n)) * spe[ix_1]
        acc_max_index = ix_1[np.argmax(acc)]
        best_point = (1 - spe[acc_max_index], sen[acc_max_index])
        auc = np.around(np.trapz(fpr, tpr), 2)

        recall_1 = sen[acc_max_index]
        recall_2 = spe[acc_max_index]
        precision_1 = (n_p * sen[acc_max_index]) / (n_p * sen[acc_max_index] + n_n * (1 - spe[acc_max_index]))
        precision_2 = (n_n * spe[acc_max_index]) / (n_n * spe[acc_max_index] + n_p * (1 - sen[acc_max_index]))

        report = {"auc": np.around(auc, 2), "acc": np.around(acc.max(), 2), "recall_1": np.around(recall_1, 2),
                  "recall_2": np.around(recall_2, 2), "precision_1": np.around(precision_1, 2),
                  "precision_2": np.around(precision_2, 2)
                  }
        _plot_optimal_cut(self.fpt_, self.tpr_, best_point)
        # if plot_point_on_ROC:
        #     p = Process(target=ClassificationPreprocessing, args=(fpr, tpr, best_point,))
        #     p.start()

        return report


def _plot_optimal_cut(fpr, tpr, best_point):
    """
    print the optimal point
    *not for users*
    """
    plt.plot(fpr, tpr)
    plt.scatter(best_point[0], best_point[1], c="red")
    plt.xlabel("1-specificity")
    plt.ylabel("sensitivity")
    plt.plot([0, 1], [0, 1], "--r")
    plt.show()


def _sigmoid(x):
    return 1 / (1 + math.exp(-x))
