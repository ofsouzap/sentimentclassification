from typing import List;
from naive_bayes_classifier import NBC;
from dataset import DataSet, DataLabel;
from example_data.imdb_data import loader as data_loader;
from metrics import accuracy, precision, recall, f_measure;

def main():

    # Load data

    data: DataSet = data_loader.load_data(
        limit=10000,
        use_progress_bar=True
    );

    # Create model

    model: NBC = NBC(
        { data_loader.CLASS_NEG, data_loader.CLASS_POS }
    );

    # Train model

    model.train(data.train_data, use_progress_bar=True);

    # Run model on test data

    predictions: List[DataLabel] = model.multi_run([item for item, _ in data.test_data]);

    # Compute metrics

    acc = accuracy(predictions, data.test_data_solutions);
    prec_pos = precision(predictions, data.test_data_solutions, data_loader.CLASS_POS);
    prec_neg = precision(predictions, data.test_data_solutions, data_loader.CLASS_NEG);
    rec_pos = recall(predictions, data.test_data_solutions, data_loader.CLASS_POS);
    rec_neg = recall(predictions, data.test_data_solutions, data_loader.CLASS_NEG);
    f1_pos = f_measure(predictions, data.test_data_solutions, data_loader.CLASS_POS);
    f1_neg = f_measure(predictions, data.test_data_solutions, data_loader.CLASS_NEG);

    print(f"""
+-----------------+
| Testing Results |
+-----------------+

Accuracy: {acc}

Precision:
\tPOS: {prec_pos}
\tNEG: {prec_neg}

Recall:
\tPOS: {rec_pos}
\tNEG: {rec_neg}

F1 (F-measure with beta=1):
\tPOS: {f1_pos}
\tNEG: {f1_neg}

""")

if __name__ == "__main__":
    main();
