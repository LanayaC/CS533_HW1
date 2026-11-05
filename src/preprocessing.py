"""Shared text tokenization for CS533 Assignment 1, Part 1.3."""

import re


_APOSTROPHES = str.maketrans(
    {
        "\u2018": "'",  # Left single quotation mark.
        "\u2019": "'",  # Right single quotation mark.
        "\u02bc": "'",  # Modifier letter apostrophe.
        "\uff07": "'",  # Full-width apostrophe.
    }
)
_ACRONYM = re.compile(r"(?<!\w)(?:[a-z]\.)+[a-z]\.?(?!\w)")
_POSSESSIVE = re.compile(r"(?<=[^\W_])'s(?![^\W_])|(?<=s)'(?![^\W_])")
_TOKEN = re.compile(r"[^\W_]+(?:'[^\W_]+)*")


def preprocess_text(text: object) -> list[str]:
    """Return lowercase word/number tokens, keeping stop words for Word2Vec.

    Non-string values (including None, NaN, and pandas.NA), empty strings,
    and punctuation-only strings produce an empty list. Numeric values are
    preserved when they occur within text; non-string scalars are not cast.

    Curly, modifier-letter, and full-width apostrophes become ASCII quotes.
    Dotted sequences of at least two single ASCII letters are joined before
    punctuation is split: U.S., N.B.A., and F.B.I. become us, nba, and fbi.
    A final period is optional. This is a spelling heuristic: e.g. becomes eg,
    and case distinctions such as US versus us are lost.

    Singular 's and plural s' possessive endings are removed. This also strips
    's contractions (it's -> it); other internal apostrophes remain (don't).
    Remaining punctuation, underscores, and hyphens separate tokens. Thus
    state-of-the-art becomes four tokens, 3.14 becomes 3 and 14, and 1,000
    becomes 1 and 000. No numeric interpretation or stop-word removal occurs.

    Tokens retain Unicode letters and digits. There is no stemming,
    lemmatization, accent normalization, or language-specific segmentation;
    combining accent marks can split words. The input is not modified.

    Examples
    --------
    >>> preprocess_text("The U.S. team's state-of-the-art plan for 2025.")
    ['the', 'us', 'team', 'state', 'of', 'the', 'art', 'plan', 'for', '2025']
    >>> preprocess_text(None)
    []
    """
    if not isinstance(text, str):
        return []

    normalized = text.translate(_APOSTROPHES).lower()
    normalized = _ACRONYM.sub(lambda match: match.group().replace(".", ""), normalized)
    normalized = _POSSESSIVE.sub("", normalized)
    return _TOKEN.findall(normalized)
