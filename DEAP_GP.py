"""
A demo of using Genetic Programming to classify MNIST hand-written digits dataset.
"""

from deap import base, creator, tools, gp
from HOF import HallOfFame
# from deap import base, creator, tools, gp
import algorithms
import selection
from dataset import get_mnist
from sklearn import metrics
from bisect import bisect_left
import operator
import numpy as np
import random

# GLOBAL CONFIG BELOW
SEED = 233333  # random seed

# hyper-params of genetic programming
POP_SIZE = 50
GEN_NO = 40
# POP_SIZE = 2
# GEN_NO = 3
INIT_MIN_DEPTH = 2
INIT_MAX_DEPTH = 6
GLOBAL_MAX_DEPTH = 10

CROSSOVER_PB = 0.7
MUTATION_PB = 0.1
TOURNAMENT_SIZE = 3

MUT_EXPR_MIN_DEPTH = 0
MUT_EXPR_MAX_DEPTH = 2

# dataset
NUM_CLASSES = 10


# GLOBAL CONFIG ABOVE
# SPECIAL FUNCTION_SET BELOW
def protective_div(lhs, rhs):
    return lhs / rhs if rhs != 0 else 0


def if_else(condition, lhs, rhs):
    return lhs if condition < 0 else rhs


# SPECIAL FUNCTION_SET ABOVE

def set_seed():
    random.seed(SEED)
    np.random.seed(SEED)


# Numpy helper function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def img_norm(x):
    return x / 255.0


def feat_norm(x):  # linearly norm into [-1, 1]
    return 2 * (x - np.min(x)) / (np.max(x) - np.min(x)) - 1


def flatten(x):
    return x.reshape(x.shape[0], -1)


class MNIST_GP:
    def __init__(self, num_feat=10, range_type='SRS'):
        self.set_creator()
        self.pset = self.get_primitive_set(num_feat)
        self.mstats = self.get_stats()
        self.toolbox = self.get_toolbox()
        # self.labels = list(np.linspace(0, 1, NUM_CLASSES, endpoint=False))  # boundaries at fixed distance[0, 0.1, 0.2, ..., 0.9] if num_class = 10
        self.labels = list(
            sorted([random.random() for _ in range(NUM_CLASSES)]))  # Static Range Selection (SRS) by default
        self.range_type = range_type
        self.tree, self.log = None, None  # tree remains None if haven't gone through `fit`

    @staticmethod
    def get_primitive_set(num_feat: int):
        # Init a strongly-typed primitive set
        pset = gp.PrimitiveSetTyped("main", [float for _ in range(num_feat)], float)
        # add your custom primitive below
        pset.addPrimitive(operator.add, [float, float], float)
        pset.addPrimitive(operator.sub, [float, float], float)
        pset.addPrimitive(operator.mul, [float, float], float)
        pset.addPrimitive(protective_div, [float, float], float)
        pset.addPrimitive(if_else, [float, float, float], float)
        pset.addEphemeralConstant('rand', lambda: random.uniform(-1, 1), float)
        # a random value that holds constant in a tree but may differs from another tree
        return pset

    @staticmethod
    def get_stats():
        stats_fit = tools.Statistics(lambda ind: ind.fitness.values[0])
        stats_size = tools.Statistics(len)
        mstats = tools.MultiStatistics(fitness=stats_fit, size=stats_size)
        mstats.register("avg", np.mean)
        mstats.register("std", np.std)
        mstats.register("min", np.min)
        mstats.register("max", np.max)
        return mstats

    def get_toolbox(self):
        toolbox = base.Toolbox()
        toolbox.register("expr", gp.genHalfAndHalf, pset=self.pset, min_=INIT_MIN_DEPTH, max_=INIT_MAX_DEPTH)
        toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.expr)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("compile", gp.compile, pset=self.pset)

        # Tournament size
        toolbox.register("select", selection.selTournament, tournsize=TOURNAMENT_SIZE)
        toolbox.register("mate", gp.cxOnePoint)
        toolbox.register("expr_mut", gp.genFull, min_=MUT_EXPR_MIN_DEPTH, max_=MUT_EXPR_MAX_DEPTH)
        toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=self.pset)

        # Max tree heights for crossover and mutation
        toolbox.decorate("mate", gp.staticLimit(key=operator.attrgetter("height"), max_value=GLOBAL_MAX_DEPTH))
        toolbox.decorate("mutate", gp.staticLimit(key=operator.attrgetter("height"), max_value=GLOBAL_MAX_DEPTH))
        return toolbox

    def get_feature(self, data):
        """
        对28x28的MNIST图像，选取以下10个作为统计特征：
        均匀划分的四个正方形区域的均值与标准差，共8个
        全图的均值与标准差，共2个
        :param data: np.ndarray [N, 784]
        :return: feats [N, 10]
        """
        feat_func = lambda x: [np.mean(x, axis=1), np.std(x, axis=1)]
        feats = []
        feats.extend(feat_func(data))  # global
        data = data.reshape(data.shape[0], 28, 28)
        feats.extend(feat_func(flatten(data[:, 0:14, 0:14])))
        feats.extend(feat_func(flatten(data[:, 15:28, 0:14])))
        feats.extend(feat_func(flatten(data[:, 0:14, 15:28])))
        feats.extend(feat_func(flatten(data[:, 15:28, 15:28])))
        return feat_norm(np.vstack(feats)).transpose().tolist()  # [N, 10]

    def get_label_by_output(self, pred_value):
        idx = bisect_left(self.labels, pred_value) - 1  # 0~9 if pred_value in (0,1)
        if idx < 0:
            return 0
        if idx > len(self.labels):
            return len(self.labels)
        return idx

    def predict_ind(self, ind, data):
        tree = self.toolbox.compile(expr=ind)
        feats = self.get_feature(img_norm(data))  # [N, 10]
        output_values = [sigmoid(tree(*feat)) for feat in feats]  # N
        # pred_labels = [self.get_label_by_output(sigmoid(tree(*feat))) for feat in feats]  # for each image
        pred_labels = [self.get_label_by_output(v) for v in output_values]  # N
        return np.asarray(pred_labels), np.asarray(output_values)

    def fitness_func(self, ind, data, gt_labels):  # should be registered under `evaluate`
        pred_labels, output_values = self.predict_ind(ind, data)
        acc = metrics.accuracy_score(gt_labels, pred_labels)
        return acc, output_values, gt_labels  # comma needed since deap treats everything as Iterable

    @staticmethod
    def set_creator():
        creator.create("FitnessMax", base.Fitness,
                       weights=(1.0, 1.0, 1.0))  # ignore output_values by setting its weight at 0
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax)

    def fit(self, X_train, Y_train):
        """
        Train from MNIST dataset
        :param X_train: [N, 784]
        :param Y_train: [N]
        :return: None, but self.tree will be recorded through tools.HallOfFame
        """
        self.toolbox.register('evaluate', self.fitness_func, data=X_train, gt_labels=Y_train)
        pop = self.toolbox.population(n=POP_SIZE)
        hof = HallOfFame(1)
        pop, log = algorithms.eaSimple(pop, self.toolbox, CROSSOVER_PB,
                                       MUTATION_PB, GEN_NO,
                                       stats=self.mstats,
                                       halloffame=hof, verbose=True,
                                       num_classes=10 if self.range_type == 'CDRS' else None)
        self.log = log
        self.tree = hof[0]

    def predict(self, X_test, Y_test=None):
        """
        Inferring labels using test-set, accuracy won't be reported if Y_test is left default.
        :param X_test: [N, 784]
        :param Y_test: [N]
        :return: pred_labels, acc (None if Y_test is None)
        """
        if self.tree is None:
            raise RuntimeError("GP haven't been trained via fit()!")
        pred_labels, _ = self.predict_ind(self.tree, X_test)
        acc = None if Y_test is None else metrics.accuracy_score(pred_labels, Y_test)
        return pred_labels, acc


def main():
    set_seed()
    X_train, Y_train, X_test, Y_test = get_mnist()
    model = MNIST_GP()

    # pool = multiprocessing.Pool()
    # model.toolbox.register('map', pool.map)

    model.fit(X_train, Y_train)
    pred_labels, acc = model.predict(X_test, Y_test)
    print(acc)


if __name__ == '__main__':
    main()
