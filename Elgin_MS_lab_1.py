# Elgin_MS_lab_1.py

def simple_probability(m: int, n: int) -> float:
    return m / n


def logical_or(m: int, k: int, n: int) -> float:
    return (m + k) / n


def logical_and(m: int, k: int, n: int, l: int) -> float:
    return (m / n) * (k / l)


def expected_value(values: tuple, probabilities: tuple) -> float:
    return sum(v * p for v, p in zip(values, probabilities))


def conditional_probability(values: tuple) -> float:
    count_a1 = 0
    count_a1b1 = 0
    for a, b in values:
        if a == 1:
            count_a1 += 1
            if b == 1:
                count_a1b1 += 1
    return count_a1b1 / count_a1


def bayesian_probability(a: float, ba: float) -> float:
    pb = a * ba + (1 - a) * (1 - ba)
    return (a * ba) / pb
    


