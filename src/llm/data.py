"""Data loading and prompt construction for LLM PPO."""
import json
import random
from typing import List, Dict, Tuple

def load_puzzles(path: str) -> List[Dict]:
    """
    Load connections.json and sanity-check:
    - each puzzle has 4 'answers'
    - exactly 16 unique words
    """
    with open(path, 'r') as f:
        puzzles = json.load(f)
        
    valid_puzzles = []
    for p in puzzles:
        # Flatten members
        all_words = []
        for group in p['answers']:
            all_words.extend(group['members'])
            
        # Sanity checks
        if len(p['answers']) != 4:
            continue
        if len(all_words) != 16:
            continue
        if len(set(all_words)) != 16:
            continue
            
        # Store words in the puzzle dict for easy access
        p['all_words'] = all_words
        valid_puzzles.append(p)
        
    print(f"Loaded {len(valid_puzzles)} valid puzzles from {len(puzzles)} total.")
    return valid_puzzles

def build_prompt(puzzle: Dict, shuffle_words: bool = True) -> Tuple[str, List[str]]:
    """
    Given a puzzle, return:
      - prompt: a string instruction + the 16 words in random order
      - words: the shuffled list of 16 words (for validation)
    """
    words = list(puzzle['all_words'])
    if shuffle_words:
        random.shuffle(words)
        
    words_str = ", ".join(words)
    
    prompt = f"""You are playing a New York Times-style “Connections” puzzle.

Rules:
- There are 16 words that form 4 hidden groups of 4 words.
- Each word must belong to exactly one group.
- Each group should represent a coherent category (e.g., fruits, sports teams, musical instruments).
- Your job is to partition ALL 16 words into 4 groups of 4, and name the categories.

Here are the 16 words (in random order):
{words_str}

Respond ONLY with a JSON object with this exact schema:

{{
  "groups": [
    {{
      "category": "<CATEGORY_NAME_1>",
      "members": ["WORD_1", "WORD_2", "WORD_3", "WORD_4"]
    }},
    {{
      "category": "<CATEGORY_NAME_2>",
      "members": ["WORD_5", "WORD_6", "WORD_7", "WORD_8"]
    }},
    {{
      "category": "<CATEGORY_NAME_3>",
      "members": ["WORD_9", "WORD_10", "WORD_11", "WORD_12"]
    }},
    {{
      "category": "<CATEGORY_NAME_4>",
      "members": ["WORD_13", "WORD_14", "WORD_15", "WORD_16"]
    }}
  ]
}}

Constraints:
- Use only the 16 words above, each exactly once.
- Do not add any extra words.
- Do not output any explanation.
- Output ONLY the JSON object."""

    return prompt, words

def get_true_groups(puzzle: Dict) -> List[List[str]]:
    """
    Returns a list of 4 lists, each containing the 4 member words.
    """
    return [group['members'] for group in puzzle['answers']]
