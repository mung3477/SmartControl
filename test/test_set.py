seeds = [1, 42, 12345]

human_subjects = ["a baby", "a child", "a woman", "a grandpa", "the spiderman"]
# [prompt, mask_prompt, focus_tokens]
human_prompts = [
    ("{subject} playing soccer", "{condition_subject} playing soccer", "playing"),
    ("{subject} doing meditation", "{condition_subject} doing meditation", "doing"),
    ("{subject} holding a guitar", "{condition_subject} holding a guitar", "holding"),
    ("{subject} riding a bike", "{condition_subject} riding a bike", "riding"),
    ("{subject} arms up", "{condition_subject} arms up", "arms"),
]

animal_subjects = ["a dog", "a cat", "a teddy", "a turtle", "a racoon"]
animal_prompts = [
    ("{subject} is running", "{condition_subject} is running", "running"),
    ("{subject} is jumping", "{condition_subject} is jumping", "jumping"),
    ("{subject} is sitting on a bench", "{condition_subject} is sitting on a bench", "sitting"),
    ("{subject} is swimming", "{condition_subject} is swimming", "swimming"),
    ("{subject} is climbing a tree", "{condition_subject} is climbing a tree", "climbing"),
]
