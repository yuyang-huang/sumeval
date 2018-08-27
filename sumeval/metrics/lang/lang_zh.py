import re
from sumeval.metrics.lang.base_lang import BaseLang, BasicElement


class LangZH(BaseLang):

    def __init__(self, tokenization='character'):
        super(LangZH, self).__init__("zh")
        self._symbol_replace = re.compile(r"[\.\!/_,$%\^\*\(\)\+\“\’\—\!。：？、，：:~@#￥&（）【】「」《》·]")

        if tokenization == 'jieba':
            import jieba
            self.tokenizer = jieba
            self.tokenize = self.tokenize_jieba
        elif tokenization == 'character':
            self.re_special_tokens = (
                re.compile(r'[+-]?(\d+(\.\d+)?|\.\d+)([eE][+-]?\d+)?'),
                re.compile(r'[A-Za-z]+'),
            )
            self.tokenize = self.tokenize_character
        elif tokenization == 'pretokenized':
            self.tokenize = self.tokenize_pretokenized
        else:
            raise ValueError(f'Unknown tokenization method: {tokenization!r}')

    def load_parser(self):
        if self._PARSER is None:
            from pyhanlp import HanLP
            self._PARSER = HanLP.parseDependency
        return self._PARSER

    def tokenize_jieba(self, text):
        _text = self._preprocess(text)
        words = [t for t in self.tokenizer.cut(_text, cut_all=False)]
        return words

    def tokenize_pretokenized(self, text):
        return text.split()

    def tokenize_character(self, text):
        text = self._preprocess(text)
        special_tokens = sorted(
            (x.start(), x.end(), x.group())
            for re_obj in self.re_special_tokens
            for x in re_obj.finditer(text)
        )

        words = []
        last_end = 0
        for start, end, token in special_tokens:
            words.extend(text[last_end:start])
            words.append(token)
            last_end = end
        words.extend(text[last_end:])
        return words

    def _preprocess(self, text):
        return self._symbol_replace.sub(" ", text)

    def parse_to_be(self, text):
        _text = self._preprocess(text)
        parsed = self.load_parser()(_text)
        bes = []
        for token in parsed.iterator():
            # print(f"{token.NAME}=({token.DEPREL})>{token.HEAD.LEMMA}")
            if token.POSTAG == "n" and token.HEAD.POSTAG in ["v", "a"]:
                be = BasicElement(token.NAME, token.HEAD.LEMMA,
                                  token.DEPREL)
                bes.append(be)
            elif token.POSTAG in ["v", "a"] and token.HEAD.POSTAG == "n":
                be = BasicElement(token.HEAD.NAME, token.LEMMA,
                                  token.DEPREL)
                bes.append(be)

        return bes
