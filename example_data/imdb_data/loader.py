from typing import List, Tuple, Callable;
from pathlib import Path;
from progress.bar import ShadyBar;
from dataset import DataSet, DataLabel, Token, TokenSequence;
import tokenizer;

TOKENIZER: Callable[[str], TokenSequence] = tokenizer.nltk_tokenize;

TRAIN_DATA_POS_PATH: Path = Path("example_data/imdb_data/reviews/training/pos").absolute();
TRAIN_DATA_NEG_PATH: Path = Path("example_data/imdb_data/reviews/training/neg").absolute();
TEST_DATA_POS_PATH: Path = Path("example_data/imdb_data/reviews/testing/pos").absolute();
TEST_DATA_NEG_PATH: Path = Path("example_data/imdb_data/reviews/testing/neg").absolute();

CLASS_POS: DataLabel = 1;
CLASS_NEG: DataLabel = 0;

def read_file(path: Path) -> str:
    return path.read_text(encoding="UTF-8");

def load_data(limit: int = -1,
    use_progress_bar: bool = False) -> DataSet:

    train_raw: List[Tuple[str, DataLabel]] = [];
    test_raw: List[Tuple[str, DataLabel]] = [];

    # Prepare to read files

    train_pos_ps: List[Path] = list(TRAIN_DATA_POS_PATH.glob("*.txt"));
    train_neg_ps: List[Path] = list(TRAIN_DATA_NEG_PATH.glob("*.txt"));
    test_pos_ps: List[Path] = list(TEST_DATA_POS_PATH.glob("*.txt"));
    test_neg_ps: List[Path] = list(TEST_DATA_NEG_PATH.glob("*.txt"));

    # Use limit if provided

    if limit >= 0:

        half_n: int = limit // 2;

        train_pos_ps = train_pos_ps[:half_n];
        train_neg_ps = train_neg_ps[:limit-half_n];
        test_pos_ps = test_pos_ps[:half_n];
        test_neg_ps = test_neg_ps[:limit-half_n];

    # Set up progress bar if requested

    tot_ps_count: int = len(train_pos_ps) + len(train_neg_ps) + len(test_pos_ps) + len(test_neg_ps);

    if use_progress_bar: read_files_progress_bar = ShadyBar("Reading reviews", max=tot_ps_count);

    # Read files

    for p in train_pos_ps:
        train_raw.append((read_file(p), CLASS_POS));
        if use_progress_bar: read_files_progress_bar.next(); # type: ignore

    for p in train_neg_ps:
        train_raw.append((read_file(p), CLASS_NEG));
        if use_progress_bar: read_files_progress_bar.next(); # type: ignore

    for p in test_pos_ps:
        test_raw.append((read_file(p), CLASS_POS));
        if use_progress_bar: read_files_progress_bar.next(); # type: ignore

    for p in test_neg_ps:
        test_raw.append((read_file(p), CLASS_NEG));
        if use_progress_bar: read_files_progress_bar.next(); # type: ignore

    if use_progress_bar: read_files_progress_bar.finish(); # type: ignore

    # Tokenize data

    train_data: List[Tuple[TokenSequence, DataLabel]] = [];
    test_data: List[Tuple[TokenSequence, DataLabel]] = [];

    tokenizing_count: int = len(train_raw) + len(test_raw);

    if use_progress_bar: tokenize_progress_bar = ShadyBar("Tokenizing reviews", max=tokenizing_count);

    for text, c in train_raw:
        train_data.append((TOKENIZER(text), c));
        if use_progress_bar: tokenize_progress_bar.next(); # type: ignore

    for text, c in test_raw:
        test_data.append((TOKENIZER(text), c));
        if use_progress_bar: tokenize_progress_bar.next(); # type: ignore

    if use_progress_bar: tokenize_progress_bar.finish(); # type: ignore

    # Return output

    return DataSet(
        train_data=train_data,
        dev_data=[],
        test_data=test_data
    );
