from typing import Dict, List, Union

from .types import Selected, SelectedMild

seeds = [1, 23, 42, 777, 12345][:1]

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

animal_subjects = ["a dog", "a dog plushie", "a cat", "a panda", "a teddy bear", "a polar bear", "a squirrel", "an elephant", "a fox", "a gorilla"][:1]
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

selected: Dict[str, List[Union[Selected, SelectedMild]]] = {
    "sig": [
        {
            "ref": "selected/Guitar.png",
            "ref_subj": "a man",
            "prompt": "{subject} is holding the guitar",
            "mask_prompt": "A man is holding the guitar",
            "focus_tokens": "holding guitar",
        },
        {
            "ref": "selected/pray.jpeg",
            "ref_subj": "a woman",
            "prompt": "{subject} is praying in a church",
            "mask_prompt": "A woman is praying in a church",
            "focus_tokens": "praying in"
        },
        {
            "ref": "selected/sword.jpeg",
            "ref_subj": "a man",
            "prompt": "{subject} is holding a sword",
            "mask_prompt": "A man is holding a sword",
            "focus_tokens": "holding sword"
        },
        {
            "ref": "selected/play tennis.jpeg",
            "ref_subj": "a girl",
            "prompt": "{subject} is playing tennis",
            "mask_prompt": "A girl is playing tennis",
            "focus_tokens": "playing tennis"
        },
        {
            "ref": "selected/riding a bike.png",
            "ref_subj": "a woman",
            "prompt": "{subject} is riding a bike",
            "mask_prompt": "A woman is riding a bike",
            "focus_tokens": "riding bike"
        },
        {
            "ref": "selected/yoga.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} is stretching it's arms left and right",
            "mask_prompt": "A man is stretching his arms left and right",
            "focus_tokens": "stretching arms"
        },
        {
            "ref": "selected/Cheer.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} with both arms gesture",
            "mask_prompt": "A man with both arms gesture",
            "focus_tokens": "both arms gesture"
        },
    ][:1],
    "mild": [
        {
            "ref": "selected/mild/apple on table.jpg",
            "ref_subj": "apple",
            "prompt": "A {subject} on a table",
            "subjects": ["apple", "strawberry", "pineapple"],
            "mask_prompt": "A apple on a table",
            "focus_tokens": "apple on"
        },
        {
            "ref": "selected/mild/car driving down.jpg",
            "ref_subj": "car",
            "prompt": "{subject} driving down a road",
            "subjects": ["car", "pick-up truck", "tractor"],
            "mask_prompt": "car driving down a road",
            "focus_tokens": "car driving"
        },
        {
            "ref": "selected/mild/cat ride bike.jpg",
            "ref_subj": "cat",
            "prompt": "A cat riding a {subject}",
            "subjects": ["bike", "motorcycle", "car"],
            "mask_prompt": "A cat riding a bike",
            "focus_tokens": "cat riding"
        },
        {
            "ref": "selected/mild/cup on desk.jpg",
            "ref_subj": "cup",
            "prompt": "A {subject} sitting on a desk",
            "subjects": ["cup", "vase", "candle"],
            "mask_prompt": "A cup on a desk",
            "focus_tokens": "cup on"
        },
        {
            "ref": "selected/mild/deer standing.jpg",
            "ref_subj": "deer",
            "prompt": "A {subject} standing in a field",
            "subjects": ["deer", "wolf", "corgi"],
            "mask_prompt": "A deer standing in a field",
            "focus_tokens": "deer standing"
        },
        {
            "ref": "selected/mild/eagle on branch.jpg",
            "ref_subj": "eagle",
            "prompt": "A {subject} on a branch",
            "subjects": ["eagle", "parrot", "duck"],
            "mask_prompt": "An eagle on a branch",
            "focus_tokens": "eagle on"
        },
        {
            "ref": "selected/mild/girl at pyramid.jpg",
            "ref_subj": "pyramid",
            "prompt": "A girl standing in front of {subject}",
            "subjects": ["pyramid", "Arch of Triumph", "Sydney Opera House", "Taj Mahal", "Colosseum in Rome"],
            "mask_prompt": "A girl standing in front of pyramid",
            "focus_tokens": "pyramid"
        },
        {
            "ref": "selected/mild/hulk in city.jpg",
            "ref_subj": "hulk",
            "prompt": "{subject} in a destroyed city",
            "subjects": ["hulk", "spiderman", "obama"],
            "mask_prompt": "hulk in a destoyed city",
            "focus_tokens": "hulk in"
        },
        {
            "ref": "selected/mild/jerry.jpg",
            "ref_subj": "jerry",
            "prompt": "{subject} at the beach",
            "subjects": ["jerry", "pikachu", "Spongebob squarepants", "Winnie the pooh", "Micky mouse"],
            "mask_prompt": "jerry at the room",
            "focus_tokens": "jerry at"
        },
        {
            "ref": "selected/mild/llama in field.jpg",
            "ref_subj": "llama",
            "prompt": "A {subject} standing in a field",
            "subjects": ["llama", "deer", "pig"],
            "mask_prompt": "A llama standing in a field",
            "focus_tokens": "llama standing"
        },
        {
            "ref": "selected/mild/pineapples.jpg",
            "ref_subj": "pineapples",
            "prompt": "Two {subject} on a towel",
            "subjects": ["pineapples", "pears", "apples"],
            "mask_prompt": "Two pineapples on a towel",
            "focus_tokens": "Two pineapples on"
        },
        {
            "ref": "selected/mild/squirrel blocks.jpg",
            "ref_subj": "squirrel",
            "prompt": "A {subject} playing with blocks",
            "subjects": ["squirrel", "hedgehog", "horse"],
            "mask_prompt": "A squirrel playing with blocks",
            "focus_tokens": "squirrel playing"
        },
    ]
}

"""
        {
            "ref": "selected/Pray.jpg",
            "prompt": "{subject} is praying",
            "mask_prompt": "A woman is praying",
            "focus_tokens": "praying"
        },
        {
            "ref": "selected/salute.jpg",
            "prompt": "{subject} is saluting",
            "mask_prompt": "A man is saluting",
            "focus_tokens": "saluting"
        },

        {
            "ref": "selected/Handstand.jpg",
            "prompt": "{subject} is doing a handstand",
            "mask_prompt": "A woman is doing a handstand",
            "focus_tokens": "doing handstand"
        },
        {
            "ref": "selected/shooting.png",
            "prompt": "{subject} is shooting a pistol in one hand",
            "mask_prompt": "A man is shooting a pistol in one hand",
            "focus_tokens": "shooting pistol"
        },
        {
            "ref": "selected/doing deadlift.png",
            "prompt": "{subject} is deadlifting",
            "mask_prompt": "A man is deadlifting",
            "focus_tokens": "deadlifting"
        },
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
