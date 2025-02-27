from typing import List, Dict
from .types import Selected

seeds = [1, 23, 42, 777, 12345]

human_subjects = ["a baby", "a child", "a woman", "a grandpa", "the spiderman"]
# [prompt, mask_prompt, focus_tokens]
"""
human_prompts = [
    ("{subject} playing soccer", "{condition_subject} playing soccer", "playing"),
    ("{subject} doing meditation", "{condition_subject} doing meditation", "doing"),
    ("{subject} holding a guitar", "{condition_subject} holding a guitar", "holding"),
    ("{subject} riding a bike", "{condition_subject} riding a bike", "riding"),
    ("{subject} arms up", "{condition_subject} arms up", "arms"),
]
"""
human_prompts = [
    ("{subject} playing soccer", "{condition_subject} playing soccer", "{condition_subject} playing soccer"),
    ("{subject} doing meditation", "{condition_subject} doing meditation", "{condition_subject} doing meditation"),
    ("{subject} holding a guitar", "{condition_subject} holding a guitar", "{condition_subject} holding guitar"),
    ("{subject} riding a bike", "{condition_subject} riding a bike", "{condition_subject} riding bike"),
    ("{subject} arms up", "{condition_subject} arms up", "{condition_subject} arms up"),
]

animal_subjects = ["a dog", "a cat", "a bear", "a fox", "a penguin", "a tiger", "an otter", "a capybara"]
"""
animal_prompts = [
    ("{subject} is running", "{condition_subject} is running", "running"),
    ("{subject} is jumping", "{condition_subject} is jumping", "jumping"),
    ("{subject} is sitting on a bench", "{condition_subject} is sitting on a bench", "sitting"),
    ("{subject} is swimming", "{condition_subject} is swimming", "swimming"),
    ("{subject} is climbing a tree", "{condition_subject} is climbing a tree", "climbing"),
]
"""
animal_prompts = [
    ("{subject} is running", "{condition_subject} is running", "{condition_subject} running"),
    ("{subject} is jumping", "{condition_subject} is jumping", "{condition_subject} jumping"),
    ("{subject} is sitting on a bench", "{condition_subject} is sitting on a bench", "{condition_subject} sitting on a bench"),
    ("{subject} is swimming", "{condition_subject} is swimming", "{condition_subject} swimming"),
    ("{subject} is climbing a tree", "{condition_subject} is climbing a tree", "{condition_subject} climbing a tree"),
]

selected: Dict[str, List[Selected]] = {
    "sig": [
        {
            "ref": "selected/doing deadlift.png",
            "prompt": "{subject} is deadlifting",
            "mask_prompt": "A man is deadlifting",
            "focus_tokens": "deadlifting"
        },
        {
            "ref": "selected/shooting.png",
            "prompt": "{subject} is shooting a pistol in one hand",
            "mask_prompt": "A man is shooting a pistol in one hand",
            "focus_tokens": "shooting pistol"
        },
        {
            "ref": "selected/yoga.jpg",
            "prompt": "{subject} is stretching it's arms left and right",
            "mask_prompt": "A man is stretching his arms left and right",
            "focus_tokens": "stretching arms"
        },
        {
            "ref": "selected/Handstand.jpg",
            "prompt": "{subject} is doing a handstand",
            "mask_prompt": "A woman is doing a handstand",
            "focus_tokens": "doing handstand"
        },
        {
            "ref": "selected/Meditate.jpg",
            "prompt": "{subject} is meditating",
            "mask_prompt": "A woman is meditating",
            "focus_tokens": "meditating"
        },
        {
            "ref": "selected/salute.jpg",
            "prompt": "{subject} is saluting",
            "mask_prompt": "A man is saluting",
            "focus_tokens": "saluting"
        },
               {
            "ref": "selected/Pray.jpg",
            "prompt": "{subject} is praying",
            "mask_prompt": "A woman is praying",
            "focus_tokens": "praying"
        }
    ]
}

"""
        {
            "ref": "selected/cheers.jpg",
            "prompt": "{subject} is toasting a cocktail to you",
            "mask_prompt": "A man is toasting a cocktail to you",
            "focus_tokens": "toasting cocktail"
        },
        {
            "ref": "selected/sword.jpeg",
            "prompt": "{subject} is holding a sword",
            "mask_prompt": "A man is holding a sword",
            "focus_tokens": "holding sword"
        },
        {
            "ref": "selected/throwing.jpeg",
            "prompt": "{subject} is throwing a spear",
            "mask_prompt": "A man is throwing a spear",
            "focus_tokens": "throwing spear"
        },
        {
            "ref": "selected/waving.jpg",
            "prompt": "{subject} is waving his hand",
            "mask_prompt": "A man is waving his hand",
            "focus_tokens": "waving hand"
        },
        {
            "ref": "selected/Cheer.jpg",
            "prompt": "{subject} with both arms gesture",
            "mask_prompt": "A man with both arms gesture",
            "focus_tokens": "both arms gesture"
        },
        {
            "ref": "selected/Guitar.png",
            "prompt": "{subject} is holding the guitar",
            "mask_prompt": "A man is holding the guitar",
            "focus_tokens": "holding guitar"
        },
        {
            "ref": "selected/Trumpet.png",
            "prompt": "{subject} is playing the trumpet",
            "mask_prompt": "A man is playing the trumpet",
            "focus_tokens": "playing trumpet"
        },
        """

"""
selected: Dict[str, List[Selected]] = {
    "sig": [
        {
            "ref": "selected/cheers.jpg",
            "prompt": "{subject} is toasting a cocktail to you",
            "mask_prompt": "A man is toasting a cocktail to you",
            "focus_tokens": "toasting"
        },
        {
            "ref": "selected/shooting.png",
            "prompt": "{subject} is shooting a handgun",
            "mask_prompt": "A man is shooting a handgun",
            "focus_tokens": "shooting handgun"
        },
        {
            "ref": "selected/yoga.jpg",
            "prompt": "{subject} is stretching it's arms",
            "mask_prompt": "A man is stretching his arms",
            "focus_tokens": "stretching arms"
        },
        {
            "ref": "selected/sword.jpeg",
            "prompt": "{subject} is holding a sword",
            "mask_prompt": "A man is holding a sword",
            "focus_tokens": "holding sword"
        },
        {
            "ref": "selected/throwing.jpeg",
            "prompt": "{subject} is throwing a spear",
            "mask_prompt": "A man is throwing a spear",
            "focus_tokens": "throwing spear"
        },
        {
            "ref": "selected/waving.jpg",
            "prompt": "{subject} is waving his hand",
            "mask_prompt": "A man is waving his hand",
            "focus_tokens": "waving"
        },
        {
            "ref": "selected/Cheer.jpg",
            "prompt": "{subject} with both arms gesture",
            "mask_prompt": "A man with both arms gesture",
            "focus_tokens": "both arms gesture"
        },
        {
            "ref": "selected/doing deadlift.png",
            "prompt": "{subject} doing deadlift",
            "mask_prompt": "A man doing deadlift",
            "focus_tokens": "deadlift"
        },
        {
            "ref": "selected/Guitar.png",
            "prompt": "{subject} is playing the guitar",
            "mask_prompt": "A man is playing the guitar",
            "focus_tokens": "playing guitar"
        },
        {
            "ref": "selected/Handstand.jpg",
            "prompt": "{subject} is doing handstand",
            "mask_prompt": "A woman is doing handstand",
            "focus_tokens": "handstand"
        },
        {
            "ref": "selected/Meditate.jpg",
            "prompt": "{subject} is doing meditation",
            "mask_prompt": "A woman is doing meditation",
            "focus_tokens": "meditation"
        },
        {
            "ref": "selected/Trumpet.png",
            "prompt": "{subject} is playing the trumpet",
            "mask_prompt": "A man is playing the trumpet",
            "focus_tokens": "playing trumpet"
        },
    ]
}
"""
