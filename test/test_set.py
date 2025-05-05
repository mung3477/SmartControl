from typing import Dict, List, Union

from .types import Selected, SelectedMild

seeds = [1, 23, 42, 777, 12345]

animal_subjects = ["a dog plushie", "a dog", "a cat", "a panda", "a teddy bear", "a polar bear", "a squirrel", "an elephant", "a fox", "a gorilla", "an owl"]

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
            "ref": "selected/play tennis.jpeg",
            "ref_subj": "a girl",
            "prompt": "{subject} is playing tennis",
            "mask_prompt": "A girl is playing tennis",
            "focus_tokens": "playing tennis"
        },
        {
            "ref": "selected/Trumpet.png",
            "ref_subj": "a man",
            "prompt": "{subject} is playing the trumpet",
            "mask_prompt": "A man is playing the trumpet",
            "focus_tokens": "playing trumpet"
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
        {
            "ref": "selected/salute.jpg",
            "prompt": "{subject} is saluting",
            "ref_subj": "a man",
            "mask_prompt": "A man is saluting",
            "focus_tokens": "saluting"
        },
        {
            "ref": "selected/Handstand.jpg",
            "prompt": "{subject} is doing a handstand",
            "ref_subj": "a woman",
            "mask_prompt": "A woman is doing a handstand",
            "focus_tokens": "doing handstand"
        },
        {
            "ref": "selected/shooting.png",
            "prompt": "{subject} is shooting a handgun",
            "ref_subj": "a man",
            "mask_prompt": "A man is shooting a handgun",
            "focus_tokens": "shooting handgun"
        },
        {
            "ref": "selected/doing deadlift.png",
            "prompt": "{subject} is deadlifting",
            "ref_subj": "a man",
            "mask_prompt": "A man is deadlifting",
            "focus_tokens": "deadlifting"
        },
        {
            "ref": "selected/cheers.jpg",
            "prompt": "{subject} is toasting a cocktail to you",
            "ref_subj": "a man",
            "mask_prompt": "A man is toasting a cocktail to you",
            "focus_tokens": "toasting cocktail"
        },
        {
            "ref": "selected/cooking.webp",
            "prompt": "{subject} is cooking vegetables",
            "ref_subj": "a man",
            "mask_prompt": "A man is cooking vegetables",
            "focus_tokens": "cooking vegetables"
        },
        {
            "ref": "selected/waving.jpg",
            "prompt": "{subject} is waving his hand",
            "ref_subj": "a man",
            "mask_prompt": "A man is waving his hand",
            "focus_tokens": "waving hand"
        },
    ],
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
