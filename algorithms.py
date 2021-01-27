import random
from bisect import bisect_left

import numpy as np
from deap import tools
from deap.algorithms import varAnd
from sklearn import metrics


def get_label_by_output(boundaries, pred_value):
    idx = bisect_left(boundaries, pred_value) - 1  # 0~9 if pred_value in (0,1)
    if idx < 0:
        return 0
    if idx > len(boundaries):
        return len(boundaries)
    return idx


def eaSimple(population, toolbox, cxpb, mutpb, ngen, stats=None,
             halloffame=None, verbose=__debug__, num_classes=None):
    """
    A modified eaSimple, allowing CDRS classification strategy
    CDRS is enabled if raw_labels is provided
    """
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)

        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # CDRS here !!!
        if num_classes is not None and len(invalid_ind) > 0:
            # fitness[0]: acc  fitness[1]: output_values  fitness[2]: gt_labels
            W_p = [ind.fitness.values[0] + 0.5 for ind in invalid_ind]
            # 1. for each class, calculate the center of the class:
            num_samples = [np.count_nonzero(invalid_ind[0].fitness.values[2] == c) for c in range(num_classes)]
            # [6000, 6000, ...,]
            num_individual = len(invalid_ind)  # considered new-born individual only
            # assert len(num_samples) == num_classes
            # num_samples[k]: Number of training samples belonging to Class k
            # For convenience, result_{p{\mu}_c} is passed through ind.fitness.values[1]
            center = np.zeros(num_classes)
            for c in range(num_classes):
                # dm = 0.
                # for p in range(num_individual):
                #     dm += num_samples[c] * W_p[p]
                dm = sum([num_samples[c] * W_p[p] for p in range(num_individual)])
                um = 0.
                # calculate `Result` for Class C
                # Result = np.zeros(num_individual, num_samples[c])
                for i in range(num_individual):
                    # num_samples = np.count_nonzero(invalid_ind[i].fitness.values[2] == c)
                    # assert num_samples == num_samples[c]
                    sample_indices = np.where(invalid_ind[i].fitness.values[2] == c)[0]
                    Result = np.zeros(sample_indices.size)
                    for j in range(sample_indices.size):
                        Result[j] = invalid_ind[i].fitness.values[1][sample_indices][j]
                        um += W_p[i] * Result[j]
                center[c] = um / dm
            center = list(sorted(center.tolist()))
            new_boundaries = [0]
            new_boundaries.extend([(center[i] + center[i + 1]) / 2 for i in range(num_classes - 1)])
            # fitness evaluation based on new boundaries
            for ind in invalid_ind:
                new_pred_labels = [get_label_by_output(new_boundaries, pred_value)
                                   for pred_value in ind.fitness.values[1]]
                ind.fitness.values = (metrics.accuracy_score(ind.fitness.values[2], new_pred_labels), 0, 0)

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


if __name__ == '__main__':
    # unittest for CDRS
    # 4 classes, 5 individuals that are already evaluated, 4 training samples
    class Fitness:
        def __init__(self, values):
            self.values = values


    class EasyIndividual:
        def __init__(self, values):
            self.fitness = Fitness(values)  # acc, output_values, gt_labels


    num_classes = 4
    num_individual = 5
    labels = [0, 0.25, 0.5, 0.75]
    invalid_ind = [EasyIndividual((0.5, np.array([0.1, 0.2, 0.6, 0.8]), np.array([0, 1, 2, 3]))),
                   EasyIndividual((0.5, np.array([0.1, 0.2, 0.6, 0.8]), np.array([0, 1, 2, 3]))),
                   EasyIndividual((0.5, np.array([0.1, 0.2, 0.6, 0.8]), np.array([0, 1, 2, 3]))),
                   EasyIndividual((0.5, np.array([0.1, 0.2, 0.6, 0.8]), np.array([0, 1, 2, 3]))),
                   EasyIndividual((0.5, np.array([0.1, 0.2, 0.6, 0.8]), np.array([0, 1, 2, 3]))),
                   ]
    # CDRS here !!!
    if num_classes is not None and len(invalid_ind) > 0:
        # fitness[0]: acc  fitness[1]: output_values  fitness[2]: gt_labels
        W_p = [ind.fitness.values[0] + 0.5 for ind in invalid_ind]
        # for each class, calculate the center of the class:
        num_samples = [np.count_nonzero(invalid_ind[0].fitness.values[2] == c) for c in
                       range(num_classes)]  # calculate gt_label distribution
        num_individual = len(invalid_ind)  # considered new-born individual only
        # assert len(num_samples) == num_classes
        # num_samples[k]: Number of training samples belonging to Class k
        # For convenience, result_{p{\mu}_c} is passed through ind.fitness.values[1]
        center = np.zeros(num_classes)
        for c in range(num_classes):
            # dm = 0.
            dm = sum([num_samples[c] * W_p[p] for p in range(num_individual)])
            um = 0.
            # calculate `Result` for Class C
            for i in range(num_individual):
                # num_samples = np.count_nonzero(invalid_ind[i].fitness.values[2] == c)
                # assert num_samples == num_samples[c]
                sample_indices = np.where(invalid_ind[i].fitness.values[2] == c)[0]
                Result = np.zeros(sample_indices.size)
                for j in range(sample_indices.size):
                    Result[j] = invalid_ind[i].fitness.values[1][sample_indices][j]
                    um += W_p[i] * Result[j]
            center[c] = um / dm
        center = list(sorted(center.tolist()))
        new_boundaries = [0]
        new_boundaries.extend([(center[i] + center[i + 1]) / 2 for i in range(num_classes - 1)])
        # fitness evaluation based on new boundaries
        for ind in invalid_ind:
            new_pred_labels = [get_label_by_output(new_boundaries, pred_value)
                               for pred_value in ind.fitness.values[1]]
            ind.fitness.values = (metrics.accuracy_score(ind.fitness.values[2], new_pred_labels), 0, 0)
