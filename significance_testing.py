from typing import List, Callable, Tuple;
from random import choices;
from dataset import DataLabel;

def paired_bootstrap_test(
    predsA: List[DataLabel],
    predsB: List[DataLabel],
    solutions: List[DataLabel],
    metric: Callable[[List[DataLabel], List[DataLabel]], float],
    b: int = int(1e4)
) -> float:
    """Calculates p-value for significance of difference between two models using paired bootstrap test.

:param predsA: the predictions that model A gave
:param predsB: the predictions that model B gave
:param solutions: the solutions to the test set
:param metric: a function taking predictions and solutions and returning a metric for them (e.g. accuracy, precision, recall etc.)
:param b: how many virtual test cases to create
:return: the p-value for if the test is significant
"""

    # TODO - change to use numpy
    # TODO - might not currently work properly. Need to test to check this

    # Calculate observed metrics

    metricAObs: float = metric(predsA, solutions);
    metricBObs: float = metric(predsB, solutions);
    deltaObs: float = metricAObs - metricBObs;

    # Count using virtual cases

    count_tot: int = 0;

    for _ in range(b):

        v_case: List[Tuple[DataLabel, DataLabel]] = choices(list(zip(predsA, predsB)), k=len(solutions));
        case_as: List[DataLabel] = [a for a, b in v_case];
        case_bs: List[DataLabel] = [b for a, b in v_case];

        a_metric: float = metric(case_as, solutions);
        b_metric: float = metric(case_bs, solutions);

        delta: float = a_metric - b_metric;

        if delta >= 2 * deltaObs:
            count_tot += 1;

    # Calculate p-value

    p_value: float = count_tot / b;

    # Return p-value

    return p_value;

# TODO - sign test
