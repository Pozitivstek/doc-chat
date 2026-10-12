"""Токенизация для поиска: нижний регистр, лёгкий стемминг (ru/en), стоп-слова."""
import re

_WORD = re.compile(r"[^\W_]+", re.UNICODE)
_CYRILLIC = re.compile(r"[а-я]")
MIN_LEN = 2
MIN_STEM = 3

STOPWORDS = frozenset("""
и в во не что он на я с со как а то все она так его но да ты к у же вы за бы по только ее мне было вот от
меня еще нет о из ему теперь когда даже ну вдруг ли если уже или ни быть был него до вас нибудь опять уж
вам ведь там потом себя ничего ей может они тут где есть надо ней для мы тебя их чем была сам чтоб без
будто чего раз тоже себе под будет ж тогда кто этот того потому этого какой совсем ним здесь этом один
почти мой тем чтобы нее сейчас были куда зачем всех никогда можно при наконец два об другой хоть после
над больше тот через эти нас про всего них какая много разве три эту моя впрочем свою этой перед иногда
лучше чуть том нельзя такой им более всегда конечно всю между это такое какие каком какого каких
the of and to in is it that for on as with was are be by this at or from an have not but they which
you all can has had were will your more what when who how about into than them its there their been
would could should also any some these those other such only own same very just
""".split())

# окончания, которые срезаются у русских слов (длинные — раньше коротких)
_RU_ENDINGS = tuple(sorted({
    "иями", "ями", "ами", "ией", "ого", "его", "ому", "ему", "ыми", "ими", "ают", "яют", "ает", "яет",
    "ует", "ешь", "ишь", "ать", "ять", "ить", "еть", "ыть", "уть", "ов", "ев", "ей", "ий", "ый", "ой",
    "ом", "ем", "ам", "ям", "ах", "ях", "ою", "ею", "ую", "юю", "ая", "яя", "ое", "ее", "ые", "ие",
    "их", "ых", "ия", "ья", "ье", "ью", "ию", "а", "я", "ы", "и", "у", "ю", "е", "о", "ь", "й",
}, key=len, reverse=True))

_EN_SUFFIXES = ("ing", "ed", "ly")


def _stem_ru(word: str) -> str:
    for ending in _RU_ENDINGS:
        if word.endswith(ending) and len(word) - len(ending) >= MIN_STEM:
            return word[: -len(ending)]
    return word


def _stem_en(word: str) -> str:
    if word.endswith("ss"):
        pass
    elif word.endswith("s") and len(word) - 1 >= MIN_STEM + 1:
        word = word[:-1]
    for suffix in _EN_SUFFIXES:
        if word.endswith(suffix) and len(word) - len(suffix) >= MIN_STEM:
            word = word[: -len(suffix)]
            break
    if word.endswith("e") and len(word) > 4:
        word = word[:-1]
    return word


def stem(word: str) -> str:
    if word.isdigit():
        return word
    if _CYRILLIC.search(word):
        return _stem_ru(word)
    return _stem_en(word)


def tokenize(text: str) -> list[str]:
    tokens = []
    for raw in _WORD.findall(text.lower().replace("ё", "е")):
        if len(raw) < MIN_LEN or raw in STOPWORDS:
            continue
        tokens.append(stem(raw))
    return tokens
