from typing import List, Union
from collections import Counter

from sumeval.metrics.lang.base_lang import BaseLang
from sumeval.metrics.lang import get_lang


class RougeCalculator:

    def __init__(self, stopwords=True, stemming=False, word_limit=-1, length_limit=-1, lang="en"):
        self.stemming = stemming
        self.stopwords = stopwords
        self.word_limit = word_limit
        self.length_limit = length_limit
        if isinstance(lang, str):
            self.lang = lang
            self._lang = get_lang(lang)
        elif isinstance(lang, BaseLang):
            self.lang = lang.lang
            self._lang = lang

    def tokenize(self, text_or_words, is_reference=False):
        """
        Tokenize a text under original Perl script manner.

        Parameters
        ----------
        text_or_words: str or str[]
            target text or tokenized words.
            If you use tokenized words, preprocessing is not applied.
            It allows you to calculate ROUGE under your customized tokens,
            but you have to pay attention to preprocessing.
        is_reference: bool
            for reference process or not

        See Also
        --------
        https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L1820
        """
        words = text_or_words

        def split(text):
            _words = self._lang.tokenize(text)
            return _words

        if self.word_limit > 0:
            if isinstance(words, str):
                words = split(words)
            words = words[:self.word_limit]
            words = self._lang.join(words)
        elif self.length_limit > 0:
            text = words
            if isinstance(text, (list, tuple)):
                text = self._lang.join(words)
            words = text[:self.length_limit]

        if isinstance(words, str):
            words = self._lang.tokenize_with_preprocess(words)

        words = [w.lower().strip() for w in words if w.strip()]

        if self.stopwords:
            words = [w for w in words if not self._lang.is_stop_word(w)]

        if self.stemming and is_reference:
            # stemming is only adopted to reference
            # https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L1416

            # min_length ref
            # https://github.com/andersjo/pyrouge/blob/master/tools/ROUGE-1.5.5/ROUGE-1.5.5.pl#L2629
            words = [self._lang.stemming(w, min_length=3) for w in words]
        return words

    def parse_to_be(self, text, is_reference=False):
        bes = self._lang.parse_to_be(text)

        def preprocess(be):
            be.head = be.head.lower().strip()
            be.modifier = be.modifier.lower().strip()
            if self.stemming and is_reference:
                be.head = self._lang.stemming(be.head, min_length=3)
                be.modifier = self._lang.stemming(be.modifier, min_length=3)

            return be

        bes = [preprocess(be) for be in bes]
        return bes

    def len_ngram(self, words, n):
        return max(len(words) - n + 1, 0)

    def ngram_iter(self, words, n):
        for i in range(self.len_ngram(words, n)):
            n_gram = words[i:i + n]
            yield tuple(n_gram)

    def count_ngrams(self, words, n):
        c = Counter(self.ngram_iter(words, n))
        return c

    def count_overlap(self, summary_ngrams, reference_ngrams):
        result = 0
        for k, v in summary_ngrams.items():
            result += min(v, reference_ngrams[k])
        return result

    def rouge_1(self, summary, references, alpha=0.5):
        return self.rouge_n(summary, references, 1, alpha)

    def rouge_2(self, summary, references, alpha=0.5):
        return self.rouge_n(summary, references, 2, alpha)

    def rouge_n(self, summary, references, n, alpha=0.5):
        """
        Calculate ROUGE-N score.

        Parameters
        ----------
        summary: str
            summary text
        references: str or str[]
            reference or references to evaluate summary
        n: int
            ROUGE kind. n=1, calculate when ROUGE-1
        alpha: float (0~1)
            alpha -> 0: recall is more important
            alpha -> 1: precision is more important
            F = 1/(alpha * (1/P) + (1 - alpha) * (1/R))

        Returns
        -------
        f1: float
            f1 score
        """
        _summary = self.tokenize(summary)
        summary_ngrams = self.count_ngrams(_summary, n)
        _refs = [references] if isinstance(references, str) else references
        matches = 0
        count_for_recall = 0
        for r in _refs:
            _r = self.tokenize(r, True)
            r_ngrams = self.count_ngrams(_r, n)
            matches += self.count_overlap(summary_ngrams, r_ngrams)
            count_for_recall += self.len_ngram(_r, n)
        count_for_prec = len(_refs) * self.len_ngram(_summary, n)
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1

    def _calc_f1(self, matches, count_for_recall, count_for_precision, alpha, w=1.):
        def safe_div(x1, x2):
            return 0 if x2 == 0 else x1 / x2
        recall = safe_div(matches, count_for_recall ** w) ** (1 / w)
        precision = safe_div(matches, count_for_precision ** w) ** (1 / w)
        denom = (1.0 - alpha) * precision + alpha * recall
        return safe_div(precision * recall, denom)

    def _lcs_table(self, reference, candidate, w=1.):
        """Create 2-d LCS score table."""
        rows = len(reference)
        cols = len(candidate)

        lcs_table = [[0] * (cols + 1) for _ in range(rows + 1)]
        consecutive_table = [[0] * (cols + 1) for _ in range(rows + 1)]  # for ROUGE-W

        for i in range(1, rows + 1):
            for j in range(1, cols + 1):
                if reference[i - 1] == candidate[j - 1]:
                    num_consecutive = consecutive_table[i - 1][j - 1]
                    consecutive_score = (num_consecutive + 1) ** w - num_consecutive ** w
                    lcs_table[i][j] = lcs_table[i - 1][j - 1] + consecutive_score
                    consecutive_table[i][j] = num_consecutive + 1
                else:
                    lcs_table[i][j] = max(lcs_table[i - 1][j], lcs_table[i][j - 1])
                    consecutive_table[i][j] = 0

        return lcs_table

    def union_lcs_ind(self, reference: List[str], candidates: List[List[str]],
                      w: float = 1.) -> List[str]:
        return sorted({i for c in candidates for i in self.lcs_ind(reference, c, w=w)})

    def lcs_ind(self, reference: List[str], candidate: List[str], w: float = 1.) -> List[int]:
        def backtrack(table):
            """Read out LCS indices."""
            i = len(reference)
            j = len(candidate)
            lcs = []
            while i > 0 and j > 0:
                if reference[i - 1] == candidate[j - 1]:
                    lcs.insert(0, i - 1)
                    i -= 1
                    j -= 1
                elif table[i][j - 1] > table[i - 1][j]:
                    j -= 1
                else:
                    i -= 1
            return lcs

        lcs_table = self._lcs_table(reference, candidate, w=w)
        return backtrack(lcs_table)

    def lcs(self, reference: List[str], candidate: List[str], w: float = 1.) -> int:
        if len(reference) == 0 or len(candidate) == 0:
            return 0

        lcs_table = self._lcs_table(reference, candidate, w=w)
        return lcs_table[-1][-1]

    def rouge_l(self, *args, **kwargs):
        return self.rouge_w(w=1., *args, **kwargs)

    def rouge_w(self, summary: str, references: Union[str, List[str]],
                alpha: float = 0.5, w: float = 1.2, summary_level: bool = False):
        """Calculate ROUGE-W score.

        To compute summary-level ROUGE-L, join the summary sentences with the
        newline character and set ``summary_level=True``.

        Parameters
        ----------
        summary : str
            summary text
        references : str or List[str]
            reference or references to evaluate summary
        alpha : float
            alpha -> 0: recall is more important
            alpha -> 1: precision is more important
            F = 1 / (alpha * (1 / P) + (1 - alpha) * (1 / R))
        w : float
            The importance factor for consecutive LCS matches. Set to 1 for ROUGE-L.
        summary_level : bool
            Whether to use summary-level evaluation instead of sentence-level.
            Defaults to False.

        Returns
        -------
        f1 : float
            The F1 score.
        """
        if isinstance(references, str):
            references = [references]
        reference_tokens = [self.tokenize(x, is_reference=True) for x in references]

        if summary_level:
            candidate_tokens = [self.tokenize(x) for x in summary.split('\n')]
        else:
            candidate_tokens = [self.tokenize(summary)]

        # Prevent double-counting
        reference_token_counter = Counter(x for tokens in reference_tokens for x in tokens)
        candidate_token_counter = Counter(x for tokens in candidate_tokens for x in tokens)

        total_reference_tokens = sum(reference_token_counter.values())
        total_candidate_tokens = sum(candidate_token_counter.values())

        matches = 0
        for ref in reference_tokens:
            if summary_level:
                num_consecutive = 0
                last_idx = None
                for idx in self.union_lcs_ind(ref, candidate_tokens, w=w):
                    token = ref[idx]
                    if reference_token_counter[token] > 0 and candidate_token_counter[token] > 0:
                        consecutive_score = (num_consecutive + 1) ** w - num_consecutive ** w
                        matches += consecutive_score

                        if last_idx is None or last_idx + 1 == idx:
                            num_consecutive += 1
                        else:
                            num_consecutive = 0
                        last_idx = idx

                        reference_token_counter[token] -= 1
                        candidate_token_counter[token] -= 1
            else:
                matches += self.lcs(ref, candidate_tokens[0], w=w)

        f1 = self._calc_f1(matches, total_reference_tokens, total_candidate_tokens, alpha, w=w)
        return f1

    def count_be(self, text, compare_type, is_reference=False):
        bes = self.parse_to_be(text, is_reference)
        be_keys = [be.as_key(compare_type) for be in bes]
        c = Counter(be_keys)
        return c

    def rouge_be(self, summary, references, compare_type="HMR", alpha=0.5):
        """
        Calculate ROUGE-BE score.

        Parameters
        ----------
        summary: str
            summary text
        references: str or str[]
            reference or references to evaluate summary
        compare_type: str
            "H", "M", "R" or these combination.
            Each character means basic element component.
            H: head, M: modifier, R: relation.
            The image of these relation is following.
            {head word}-{relation}->{modifier word}
            When "HMR", use head-relation-modifier triple as basic element.
        alpha: float (0~1)
            alpha -> 0: recall is more important
            alpha -> 1: precision is more important
            F = 1/(alpha * (1/P) + (1 - alpha) * (1/R))

        Returns
        -------
        f1: float
            f1 score
        """
        matches = 0
        count_for_recall = 0
        s_bes = self.count_be(summary, compare_type)
        _refs = [references] if isinstance(references, str) else references
        for r in _refs:
            r_bes = self.count_be(r, compare_type, True)
            matches += self.count_overlap(s_bes, r_bes)
            count_for_recall += sum(r_bes.values())
        count_for_prec = len(_refs) * sum(s_bes.values())
        f1 = self._calc_f1(matches, count_for_recall, count_for_prec, alpha)
        return f1
