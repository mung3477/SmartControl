from typing import TypedDict, List


class AttnDiffTrgtTokens(TypedDict):
    cond: List[str]
    gen: List[str]
