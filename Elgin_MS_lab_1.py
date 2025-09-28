# Elgin_MS_lab_1.py

def simple_probability(m: int, n: int) -> float:
    return m / n


def logical_or(m: int, k: int, n: int) -> float:
    return (m + k) / n


def logical_and(m: int, k: int, n: int, l: int) -> float:
    return (m / n) * (k / l)


def expected_value(values: tuple, probabilities: tuple) -> float:
    return sum(v * p for v, p in zip(values, probabilities))


def conditional_probability(values):
    count_a = 0
    count_a_and_b = 0
    for pair in values:
        first, second = pair
        if first == 1:
            count_A += 1
            if second == 1:
                count_a_and_b += 1
    return count_a_and_b / count_a


def bayesian_probability(a: float, ba: float) -> float:
    pb = a * ba + (1 - a) * (1 - ba)
    return (a * ba) / pb
    


