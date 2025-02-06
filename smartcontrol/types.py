from typing import List, TypedDict


class AttnDiffTrgtTokens(TypedDict):
    cond: List[str]
    gen: List[str]
