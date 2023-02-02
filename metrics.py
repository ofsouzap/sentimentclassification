from typing import List;
from dataset import DataLabel;

def accuracy(predictions: List[DataLabel], solutions: List[DataLabel]) -> float:

    assert len(predictions) == len(solutions), "Predictions and solutions of different lengths";

    correct_count: int = 0;

    for pred, sol in zip(predictions, solutions):
        if pred == sol:
            correct_count += 1;

    return correct_count / len(predictions);

def precision(predictions: List[DataLabel], solutions: List[DataLabel], c: DataLabel) -> float:

    assert len(predictions) == len(solutions), "Predictions and solutions of different lengths";

    correct_count: int = 0;
    c_pred_count: int = 0;

    for pred, sol in zip(predictions, solutions):
        if pred == c:
            c_pred_count += 1;
            if pred == sol:
                correct_count += 1;

    assert c_pred_count > 0, "None of the predictions are of the required predicted class";

    return correct_count / c_pred_count;

def recall(predictions: List[DataLabel], solutions: List[DataLabel], c: DataLabel) -> float:

    assert len(predictions) == len(solutions), "Predictions and solutions of different lengths";

    correct_count: int = 0;
    c_sol_count: int = 0;

    for pred, sol in zip(predictions, solutions):
        if sol == c:
            c_sol_count += 1;
            if pred == sol:
                correct_count += 1;

    assert c_sol_count > 0, "None of the solutions are of the required solution class";

    return correct_count / c_sol_count;

def f_measure(predictions: List[DataLabel], solutions: List[DataLabel], c: DataLabel, beta: float = 1) -> float:

    beta_sqr = beta * beta;

    prec = precision(predictions, solutions, c);
    rec = recall(predictions, solutions, c);

    # F_beta = ( (b + 1) * P * R ) / ( P * b^2 + R )
    #     b - beta
    #     P - precision
    #     R - recall

    return ( (beta_sqr + 1) * prec * rec ) / \
        ( ( beta_sqr * prec ) + rec )
