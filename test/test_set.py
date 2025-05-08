from typing import Dict, List, Union

from .types import Selected, SelectedMild

seeds = [1, 23, 42, 777, 12345]

animal_subjects = ["a dog plushie", "a dog", "a cat", "a panda", "a teddy bear", "a polar bear", "a squirrel", "an elephant", "a fox", "a gorilla", "an owl"]

selected: Dict[str, List[Union[Selected, SelectedMild]]] = {
    "sig": [
        {
            "ref": "selected/significant_conflict/a man is holding the guitar.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} is holding the guitar",
            "mask_prompt": "A man is holding the guitar",
            "focus_tokens": "holding guitar",
        },
        {
            "ref": "selected/significant_conflict/a girl is playing tennis.jpg",
            "ref_subj": "a girl",
            "prompt": "{subject} is playing tennis",
            "mask_prompt": "A girl is playing tennis",
            "focus_tokens": "playing tennis"
        },
        {
            "ref": "selected/significant_conflict/a man is playing the trumpet.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} is playing the trumpet",
            "mask_prompt": "A man is playing the trumpet",
            "focus_tokens": "playing trumpet"
        },
        {
            "ref": "selected/significant_conflict/a woman is praying in a church.jpg",
            "ref_subj": "a woman",
            "prompt": "{subject} is praying in a church",
            "mask_prompt": "a woman is praying in a church",
            "focus_tokens": "praying in"
        },
        {
            "ref": "selected/significant_conflict/a man is holding a sword.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} is holding a sword",
            "mask_prompt": "a man is holding a sword",
            "focus_tokens": "holding sword"
        },
        {
            "ref": "selected/significant_conflict/a woman is riding a bike.jpg",
            "ref_subj": "a woman",
            "prompt": "{subject} is riding a bike",
            "mask_prompt": "a woman is riding a bike",
            "focus_tokens": "riding bike"
        },
        {
            "ref": "selected/significant_conflict/a man is stretching it's arms left and right.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} is stretching it's arms left and right",
            "mask_prompt": "a man is stretching his arms left and right",
            "focus_tokens": "stretching arms"
        },
        {
            "ref": "selected/significant_conflict/a man with both arms gesture.jpg",
            "ref_subj": "a man",
            "prompt": "{subject} with both arms gesture",
            "mask_prompt": "a man with both arms gesture",
            "focus_tokens": "both arms gesture"
        },
        {
            "ref": "selected/significant_conflict/a man is saluting.jpg",
            "prompt": "{subject} is saluting",
            "ref_subj": "a man",
            "mask_prompt": "a man is saluting",
            "focus_tokens": "saluting"
        },
        {
            "ref": "selected/significant_conflict/a woman is doing a handstand.jpg",
            "prompt": "{subject} is doing a handstand",
            "ref_subj": "a woman",
            "mask_prompt": "a woman is doing a handstand",
            "focus_tokens": "doing handstand"
        },
        {
            "ref": "selected/significant_conflict/a man is shooting a handgun.jpg",
            "prompt": "{subject} is shooting a handgun",
            "ref_subj": "a man",
            "mask_prompt": "a man is shooting a handgun",
            "focus_tokens": "shooting handgun"
        },
        {
            "ref": "selected/significant_conflict/a man is deadlifting.jpg",
            "prompt": "{subject} is deadlifting",
            "ref_subj": "a man",
            "mask_prompt": "a man is deadlifting",
            "focus_tokens": "deadlifting"
        },
        {
            "ref": "selected/significant_conflict/a man is toasting a cocktail to you.jpg",
            "prompt": "{subject} is toasting a cocktail to you",
            "ref_subj": "a man",
            "mask_prompt": "a man is toasting a cocktail to you",
            "focus_tokens": "toasting cocktail"
        },
        {
            "ref": "selected/significant_conflict/a man is cooking vegetables.jpg",
            "prompt": "{subject} is cooking vegetables",
            "ref_subj": "a man",
            "mask_prompt": "a man is cooking vegetables",
            "focus_tokens": "cooking vegetables"
        },
        {
            "ref": "selected/significant_conflict/a man is waving his hand.jpg",
            "prompt": "{subject} is waving his hand",
            "ref_subj": "a man",
            "mask_prompt": "a man is waving his hand",
            "focus_tokens": "waving hand"
        },
    ],
    "mild": [
        {
            "ref": "selected/mild_conflict/A apple on a table.jpg",
            "ref_subj": "apple",
            "prompt": "A {subject} on a table",
            "subjects": ["apple", "strawberry", "pineapple"],
            "mask_prompt": "A apple on a table",
            "focus_tokens": "apple on"
        },
        {
            "ref": "selected/mild_conflict/car driving downa road.jpg",
            "ref_subj": "car",
            "prompt": "{subject} driving down a road",
            "subjects": ["car", "pick-up truck", "tractor"],
            "mask_prompt": "car driving down a road",
            "focus_tokens": "car driving"
        },
        {
            "ref": "selected/mild_conflict/A cat riding a bike.jpg",
            "ref_subj": "bike",
            "prompt": "A cat riding a {subject}",
            "subjects": ["bike", "motorcycle", "car"],
            "mask_prompt": "A cat riding a bike",
            "focus_tokens": "cat riding"
        },
        {
            "ref": "selected/mild_conflict/A cup sitting on a desk.jpg",
            "ref_subj": "cup",
            "prompt": "A {subject} sitting on a desk",
            "subjects": ["cup", "vase", "candle"],
            "mask_prompt": "A cup on a desk",
            "focus_tokens": "cup on"
        },
        {
            "ref": "selected/mild_conflict/A deer standing in a field.jpg",
            "ref_subj": "deer",
            "prompt": "A {subject} standing in a field",
            "subjects": ["deer", "wolf", "corgi"],
            "mask_prompt": "A deer standing in a field",
            "focus_tokens": "deer standing"
        },
        {
            "ref": "selected/mild_conflict/A eagle on a branch.jpg",
            "ref_subj": "eagle",
            "prompt": "A {subject} on a branch",
            "subjects": ["eagle", "parrot", "duck"],
            "mask_prompt": "An eagle on a branch",
            "focus_tokens": "eagle on"
        },
        {
            "ref": "selected/mild_conflict/A girl standing in front of pyramid.jpg",
            "ref_subj": "pyramid",
            "prompt": "A girl standing in front of {subject}",
            "subjects": ["pyramid", "Arch of Triumph", "Sydney Opera House", "Taj Mahal", "Colosseum in Rome"],
            "mask_prompt": "A girl standing in front of pyramid",
            "focus_tokens": "pyramid"
        },
        {
            "ref": "selected/mild_conflict/hulk in a destroyed city.jpg",
            "ref_subj": "hulk",
            "prompt": "{subject} in a destroyed city",
            "subjects": ["hulk", "spiderman", "obama"],
            "mask_prompt": "hulk in a destroyed city",
            "focus_tokens": "hulk in"
        },
        {
            "ref": "selected/mild_conflict/jerry at the beach.jpg",
            "ref_subj": "jerry",
            "prompt": "{subject} at the beach",
            "subjects": ["jerry", "pikachu", "Spongebob squarepants", "Winnie the pooh", "Micky mouse"],
            "mask_prompt": "jerry at the room",
            "focus_tokens": "jerry at"
        },
        {
            "ref": "selected/mild_conflict/A llama standing in a field.jpg",
            "ref_subj": "llama",
            "prompt": "A {subject} standing in a field",
            "subjects": ["llama", "deer", "pig"],
            "mask_prompt": "A llama standing in a field",
            "focus_tokens": "llama standing"
        },
        {
            "ref": "selected/mild_conflict/Two pineapples on a towel.jpg",
            "ref_subj": "pineapples",
            "prompt": "Two {subject} on a towel",
            "subjects": ["pineapples", "pears", "apples"],
            "mask_prompt": "Two pineapples on a towel",
            "focus_tokens": "Two pineapples on"
        },
        {
            "ref": "selected/mild_conflict/A squirrel playing with blocks.jpg",
            "ref_subj": "squirrel",
            "prompt": "A {subject} playing with blocks",
            "subjects": ["squirrel", "hedgehog", "horse"],
            "mask_prompt": "A squirrel playing with blocks",
            "focus_tokens": "squirrel playing"
        },
    ]
}
