from enum import Enum
import os
import json
import random
import string
import typing

from tabulate import tabulate

from prompts import WELCOME_MESSAGE

class PuzzleReponse(Enum):
    INVALID = 0
    INCORRECT = 1
    NEARLY_CORRECT = 2
    CORRECT = 3

class ConnectionsPuzzle():
    '''
    An instance of a "Connections" puzzle taken from the NYT archive. The puzzle
    consists of a list of 16 words grouped into four (hidden) categories. To play,
    the user must select 4 words that they believe belong to the same category. If
    they are correct, the category is revealed. The puzzle proceeds until every 
    category has been revealed or the user runs out of guesses. In the optional
    'all-in-one' mode, the user must guess all categories at once.

    Args:
        id (int): The puzzle ID number, from 1 to 150
        num_guesses (int): The number of guesses the user is allowed to make
        all_in_one (bool): Whether the user needs to guess all categories at once
        data_dir (str): The directory where the puzzle data is stored
    '''
    def __init__(self,
                 id: int,
                 num_guesses: int = 4,
                 all_in_one: bool = False,
                 data_dir: str = "./data"):
        
        self.data = self._load_data(id, data_dir)
        self.num_guesses = num_guesses
        self.all_in_one = all_in_one

    def _load_data(self, id: int, data_dir: str) -> dict:
        '''
        Load the puzzle data from a JSON file.
        '''
        with open(os.path.join(data_dir, "puzzle_data.json"), "r") as f:
            data = json.load(f)
            
        return data[id-1]
    
    def _format(self, word: str) -> str:
        '''
        Format a word by removing punctuation and spaces, and converting it to uppercase 
        '''
        word = word.translate(str.maketrans('', '', string.punctuation))
        word = word.strip()
        word = word.upper()

        return word

    
    def render(self, observation: dict) -> str:
        '''
        Render the puzzle state as a string
        '''
        
        if observation["message"] == "":
            display = WELCOME_MESSAGE

        else:
            display = observation["message"]
        
        # Display successfully guessed categories
        for category in observation["revealed"]:
            display += f"\n\n{category['description']}: {', '.join(category['words'])}"

        # Render the words
        word_grid = tabulate([observation["words"][i:i+4] for i in range(0, len(observation["words"]), 4)],
                             tablefmt="simple_grid")
        
        display += f"\n\n{word_grid}"

        display += f"\n\nYou have {observation['guesses_remaining']} incorrect guesses reamining."

        return display

    def reset(self) -> dict:
        '''
        Reset the puzzle to its initial state and return the initial observation
        '''
        self.guesses = []
        self.revealed = []
        self.revealed_colors = []
        self.guesses_remaining = self.num_guesses
        

        self.words = sum([category["words"] for category in self.data['answers']], [])
        self.words = [self._format(word) for word in self.words]
        random.shuffle(self.words)

        observation = {
            "words": self.words,
            "guesses": self.guesses,
            "revealed": self.revealed,
            "revealed_colors": self.revealed_colors,
            "guesses_remaining": self.guesses_remaining,
            "all_in_one": self.all_in_one,
            "message": "",
            "response": PuzzleReponse.INVALID
        }


        done = False
        reward = 0

        return observation, done, reward
    
    def get_difficulty(self, category):
        color_dict = {'#df7bea': 'purple',
                      '#fbd400': 'yellow',
                      '#69e352': 'green',
                      '#5492ff': 'blue'}
        color_hex_str = category['color']
        return color_dict[color_hex_str]




    def step(self, action: typing.Union[typing.List[str], typing.List[typing.List[str]]]) -> dict:
        '''
        Apply the user's action and return the updated observation
        '''

        if not self.all_in_one:
            action = [self._format(word) for word in action]
            valid = all([(word in self.words) for word in action]) and len(action) == 4
        
        else:
            action = [[self._format(word) for word in category] for category in action]
            valid = all([all([(word in self.words) for word in category]) and len(category) == 4 for category in action])

        # Discard invalid actions
        if not valid:
            observation = {
                "words": self.words,
                "guesses": self.guesses,
                "revealed": self.revealed,
                "revealed_colors": self.revealed_colors,
                "guesses_remaining": self.guesses_remaining,
                "all_in_one": self.all_in_one,
                "message": "Invalid guess. Please try again.",
                "response": PuzzleReponse.INVALID
            }

            done = False
            reward = 0

            return observation, done, reward

        # All-in-one mode
        if self.all_in_one:
        
            all_match = True
            for category in self.data['answers']:
                formatted_category = [self._format(word) for word in category["words"]]

                match = False
                for category_guess in action:
                    overlap = set(category_guess).intersection(set(formatted_category))
                
                    if len(overlap) == 4:
                        match = True
                        break

                if not match:
                    all_match = False
                    break

            if all_match:
                self.revealed = self.data
                self.revealed_colors = [self.get_difficulty(category) for category in self.data['answers']]
                self.words = []

                message = "Correct! You guessed all categories."
                response = PuzzleReponse.CORRECT

            else:
                message = "Incorrect guess."
                response = PuzzleReponse.INCORRECT
                self.guesses_remaining -= 1

        # Iterative mode
        else:
            self.guesses.append(action)

            # Check correctness
            correct_category = None
            off_by_one = False
            for category in self.data['answers']:
                formatted_category = [self._format(word) for word in category["words"]]
                overlap = set(action).intersection(set(formatted_category))
                
                if len(overlap) == 4:
                    correct_category = category
                    break
                
                elif len(overlap) == 3:
                    off_by_one = True

            if correct_category is not None:
                self.revealed.append(correct_category)
                self.words = [word for word in self.words if word not in action]
                self.revealed_colors.append(self.get_difficulty(correct_category))

                message = f"Correct! The category was {correct_category['description']}. Diffulty: {self.get_difficulty(correct_category)}."
                response = PuzzleReponse.CORRECT

            elif off_by_one:
                message = "Nearly Correct. Three of your words are in a group, but one is not in the same group."
                response = PuzzleReponse.NEARLY_CORRECT
                self.guesses_remaining -= 1
        
            else:
                message = "Incorrect guess."
                response = PuzzleReponse.INCORRECT
                self.guesses_remaining -= 1

        # Done when we've revealed at least 3 categories, since the last group is guaranteed
        done = (self.guesses_remaining == 0) or (len(self.revealed) >= 3)
        reward = 1 if len(self.revealed) >= 3 else 0

        if reward == 1:
            self.revealed = self.data
            self.revealed_colors = [self.get_difficulty(category) for category in self.data['answers']]
            self.words = []

        # Update the observation
        observation = {
            "words": self.words,
            "guesses": self.guesses,
            "revealed": self.revealed,
            "revealed_colors": self.revealed_colors,
            "guesses_remaining": self.guesses_remaining,
            "all_in_one": self.all_in_one,
            "message": message,
            "response": response
        }

        return observation, done, reward