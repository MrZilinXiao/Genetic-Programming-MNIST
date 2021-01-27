from deap.tools import selRandom


def selTournament(individuals, k, tournsize):
    chosen = []
    for i in range(k):
        aspirants = selRandom(individuals, tournsize)
        chosen.append(max(aspirants, key=lambda x: x.fitness.values[0]))
    return chosen
