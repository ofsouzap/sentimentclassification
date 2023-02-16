from significance_testing import paired_bootstrap_test;
from metrics import precision, accuracy;
from dataset import DataLabel;
from typing import List;

predsA: List[DataLabel] =   [0]*50 + [1]*50;
predsB: List[DataLabel] =   [1]*25 + [0]*25 + [1]*50;
sols: List[DataLabel] =     [0]*50 + [1]*50;

# TODO - sometime, use this to help check that the bootstrap test has been implemented properly

# predsA = [1, 1, 1, 0, 1, 0, 1, 1, 0, 1];
# predsB = [1, 0, 1, 1, 0, 1, 0, 1, 0, 0];
# sols = [1] * 10;

print(paired_bootstrap_test(
    predsA,
    predsB,
    sols,
    lambda preds, sols: accuracy(preds, sols),#precision(preds, sols, 1),
    int(1e4)
));
