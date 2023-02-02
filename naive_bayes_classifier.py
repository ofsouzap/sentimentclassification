from typing import List, Optional, Dict, Tuple;
from dataset import DataLabel, Token, TokenSequence;
from model import Model, RunningUntrainedModelException;
from math import log;

class NBC(Model):

    def __init__(self,
        cs: set[DataLabel],
        smoothing_factor: int = 1,
        use_training_assertions: bool = True):


        # Set of possible classes to output
        self.cs = cs;

        # Amount to start occurences of tokens with
        self.smoothing_factor = smoothing_factor;

        # Whether to run assertions after/during training. Will be slower but safer
        self.use_training_assertions = use_training_assertions;


        # Whether the model has been trained yet
        self._trained: bool = False;

        # self._log_class_probabilities[c] = log P(c)
        self._log_class_probabilities: Dict[DataLabel, float] = {};

        # self._log_token_probabilities[c][w] = log P(w|c)
        self._log_token_probabilities: Dict[DataLabel, Dict[Token, float]] = {};

        # Any tokens that the model was trained with. Tokens are ignore if not in the vocabulary
        self._vocabulary: set[Token] = set();

    @property
    def is_trained(self) -> bool:
        return self._trained;

    def train(self,
        data: List[Tuple[TokenSequence, DataLabel]],
        use_progress_bar: bool = False
    ) -> None:

        # Approximating metrics being calculated:
        #     P(c) ~= (number of c data items) / (total number of data items)
        #     P(t|c) ~=
        #         (number of t tokens in items of class c) + (smoothing factor)
        #             /
        #         (total number of tokens in items of class c) + (smoothing factor)

        # N.B. uses add-alpha smoothing where alpha is self.smoothing_factor

        ###############
        # Preparation #
        ###############

        # Progress bar init
        if use_progress_bar: self.train_progress_init(len(data));

        # Initialize vocabulary

        self._vocabulary = set();

        # Initialize class counts

        class_occurences: Dict[DataLabel, int] = {};
        for c in self.cs: class_occurences[c] = 0;

        # Initialize token counts

        token_occurences: Dict[DataLabel, Dict[Token, int]] = {};
        for c in self.cs: token_occurences[c] = {};

        ###################
        # Processing Data #
        ###################

        # Analyze data

        for tokens, c in data:

            # Class count data

            if c not in class_occurences:
                class_occurences[c] = self.smoothing_factor;

            class_occurences[c] += 1;

            # Go through tokens

            for token in tokens:

                # Add class this token occurence

                if token not in token_occurences[c]:
                    # If token hasn't yet been seen, add to vocabulary and initialize occurences counts for *all* classes
                    self._vocabulary.add(token);
                    for c1 in self.cs: token_occurences[c1][token] = self.smoothing_factor;

                token_occurences[c][token] += 1;

            # Progress bar next
            if use_progress_bar: self.train_progress_next();

        # Progress bar finish
        if use_progress_bar: self.train_progress_finish();

        ########################
        # Utility calculations #
        ########################

        # Calculate total token occurences for classes

        class_total_token_occurences: Dict[DataLabel, int] = {};

        for c in self.cs:
            tot: int = 0;
            for token in token_occurences[c]:
                tot += token_occurences[c][token];
            class_total_token_occurences[c] = tot;

        #############################
        # Probabilities calculation #
        #############################

        # Calculate log P(c) for classes

        self._log_class_probabilities = {};

        for c in self.cs:

            # log ( N_c / N ) = log N_c - log N
            #     N_c is number of items with class c
            #     N is total number of items

            self._log_class_probabilities[c] = \
                log(class_occurences[c]) - log(len(data));

        # Calculate P(t|c) for tokens in vocabulary

        self._log_token_probabilities = {};
        for c in self.cs: self._log_token_probabilities[c] = {};

        for token in self._vocabulary:

            # log ( Nt_t,c / Nt_c ) = log Nt_t,c - log Nt_c
            #     Nt_t,c is number of occurences of token t in items of class c
            #     Nt_c is total number of tokens in all items of class c

            for c in self.cs:

                self._log_token_probabilities[c][token] = \
                    log(token_occurences[c][token]) - log(class_total_token_occurences[c]);

        #############
        # Finishing #
        #############

        # Set is trained flag

        self._trained = True;

        # Verify calculated values' integrity

        if self.use_training_assertions:
            self.__assert_training_valid();

    def run(self, tokens: TokenSequence) -> DataLabel:

        # Check model is trained

        if not self.is_trained:
            raise RunningUntrainedModelException();

        # Calculate token log probabilities sums

        token_log_probabilities_sums: Dict[DataLabel, float] = {};
        for c in self.cs: token_log_probabilities_sums[c] = 0;

        for token in tokens:

            if token not in self._vocabulary:
                continue;

            for c in self.cs:

                token_log_prob: float = self._log_token_probabilities[c][token];
                token_log_probabilities_sums[c] += token_log_prob;

        # Calculate total probabilities

        tot_log_probs: Dict[DataLabel, float] = {};

        for c in self.cs:
            tot_log_probs[c] = self._log_class_probabilities[c] + token_log_probabilities_sums[c];

        # Find most probable class

        best_c: Optional[DataLabel] = None;
        best_log_prob: Optional[float] = None;

        for c in self.cs:

            log_prob = tot_log_probs[c];

            if (best_log_prob == None) or (log_prob > best_log_prob):
                best_c = c;
                best_log_prob = log_prob;

        assert best_c != None;

        # Return most probable class

        return best_c;

    def __assert_training_valid(self) -> None:

        # Check used classes are consistent

        class_prob_classes: set[DataLabel] = set();
        token_prob_classes: set[DataLabel] = set();

        for c in self._log_class_probabilities:
            class_prob_classes.add(c);

        for c in self._log_token_probabilities:
            token_prob_classes.add(c);

        assert class_prob_classes == self.cs;
        assert token_prob_classes == self.cs;

        # Check used tokens are consistent

        token_prob_tokens: Dict[DataLabel, set[Token]] = {};

        for c in self.cs:

            token_prob_tokens[c] = set();

            for token in self._log_token_probabilities[c]:
                token_prob_tokens[c].add(token);

        for c in token_prob_tokens:
            assert token_prob_tokens[c] == self._vocabulary;
