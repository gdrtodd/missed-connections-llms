WELCOME_MESSAGE = """
Welcome to the Connections puzzle! The puzzle consists of 16 words and your objective is to find groups of 4 items that share something in common.
For example, some categories and the words contained in them might be: 
'FISH': Bass, Flounder, Salmon, Trout
'FIRE ___': Ant, Drill, Island, Opal

Categories will always be more specific than, for example, '5-LETTER WORDS' or 'NAMES'

To play, select four words that you think belong to the same category and input them as a list separated by commas. For example, you could input 'Bass, Flounder, Salmon, Trout.' You do not need to guess or identify the category, only the words within it.

You will be told whether your guess is correct. If it is correct, the category will be revealed. You will also be told if three of your selected words belong to the same category, but not which three.

The game ends when you have correctly indentified all four groups or when you make too many incorrect guesses (number provided below). Good luck!
"""

SYSTEM_PROMPT = '''
You are a puzzle-solving agent, PuzzleGPT, and your objective is to solve linguistic puzzles that rely on your knowledge of words and the world according to the instructions provided by the user.
'''

INITIAL_PROMPT_ITERATIVE = '''
I want you to solve a daily word puzzle that finds commonalities between words. There are 16 words, which form 4 groups of 4 words. Each group has some common theme that links the words. You must use each of the 16 words, and use each word only once.
    
Each group of 4 words are linked together in some way. The connection between words can be simple. An example of a simple connection would be "types of fish": Bass, Flounder, Salmon, Trout. Categories can also be more complex, and require abstract or lateral thinking.
An example of this type of connection would be "things that start with FIRE": Ant, Drill, Island, Opal.

Provide the one group you are most sure of as your final answer. I will enter this into the puzzle and give you feedback: I will tell you whether it is correct, incorrect, or nearly correct (3/4 words).
Then we will continue until the puzzled is solved, or you lose.

Format your final answer as:
<ANSWER> GROUP NAME: [WORD, WORD, WORD, WORD] </ANSWER>

Some rules:
{1}- Give your final answer in the format described above (surrounded by <ANSWER> delimiters) without any additional text
- Use the message history to make sure you don't repeat any of your previous guesses

Here are the starting 16 words:
{0}
'''

COT_PROMPT_ITERATIVE = '''
- First, briefly summarize the rules and objective of the puzzle (in no more than 50 words)
- Next, come up with a category to which four of the words belong and briefly explain why you think they belong to that category
'''

INITIAL_PROMPT_ONESHOT = '''
I want you to solve a daily word puzzle that finds commonalities between words. There are 16 words, which form 4 groups of 4 words. Each group has some common theme that links the words. You must use each of the 16 words, and use each word only once.
Each group of 4 words are linked together in some way. The connection between words can be simple. An example of a simple connection would be "types of fish": Bass, Flounder, Salmon, Trout. Categories can also be more complex, and require abstract or lateral thinking.
An example of this type of connection would be "things that start with FIRE": Ant, Drill, Island, Opal.

Format your final answers as:
<ANSWER>
GROUP 1 NAME: [WORD, WORD, WORD, WORD]
GROUP 2 NAME: [WORD, WORD, WORD, WORD]
GROUP 3 NAME: [WORD, WORD, WORD, WORD]
GROUP 4 NAME: [WORD, WORD, WORD, WORD]
</ANSWER>

Replace each GROUP NAME with a name for the group you create.

Some rules:
{1}- Give your final answers in the format described above (surrounded by <ANSWER> delimiters) without any additional text
- Use the message history to make sure you don't repeat any of your previous guesses

Here are the starting 16 words:
{0}
'''

COT_PROMPT_ONESHOT = '''
- First, briefly summarize the rules and objective of the puzzle (in no more than 50 words)
- Next, come up with the four categories to which the words belong. For each category, briefly explain why each of the words you selected belong to that category
'''

THEIR_PROMPT_ONESHOT = '''
Find 4 groups, each of 4 words that share something in common, out of 16 words. I want to use them to solve a daily word puzzle that finds commonalities between words. The game is a new puzzle featured in The New York Times, inspired by crosswords. You have to use all those 16 words I give you and each word only once.
Format your answer as:
<GROUP NAME>: WORD, WORD, WORD, WORD
Do not add any additional text to your final answer, formatting is very important. Put each group on a separate line
Below are my 16 words:
{0}
'''

CORRECT_GUESS_PROMPT_ITERATIVE = '''
The response from the game was: {1}

Good job! Continue to solve the puzzle. 

Format your answer as:
<ANSWER> GROUP NAME: [WORD, WORD, WORD, WORD] </ANSWER>

As a reminder:
{2}- Give your final answer in the format described above (surrounded by <ANSWER> delimiters) without any additional text
- Use the message history to make sure you don't repeat any of your previous guesses

Here are the remaining words:
{0}
'''

NEARLY_CORRECT_GUESS_PROMPT_ITERATIVE = '''
The response from the game was: {1}

Continue to solve the puzzle. Again, provide one group you are most certain of and make sure you don't repeat any of your previous guesses.

Format your answer as:
<ANSWER> GROUP NAME: [WORD, WORD, WORD, WORD] </ANSWER>

As a reminder:
{2}- Give your final answer in the format described above (surrounded by <ANSWER> delimiters) without any additional text
- Use the message history to make sure you don't repeat any of your previous guesses

Here are the remaining words:
{0}
'''

INCORRECT_GUESS_PROMPT_ITERATIVE = '''
The response from the game was: {1}

Let's continue to solve the puzzle. Again, provide one group you are most certain of and make sure you don't repeat any of your previous guesses.

Format your answer as:
<ANSWER> GROUP NAME: [WORD, WORD, WORD, WORD] </ANSWER>

As a reminder:
{2}- Give your final answer in the format described above (surrounded by <ANSWER> delimiters) without any additional text
- Use the message history to make sure you don't repeat any of your previous guesses

Here are the remaining words:
{0}
'''

INVALID_GUESS_PROMPT_ITERATIVE = '''
The response from the game was: {1}

Your answer wasn't formatted correctly. Try again, and follow the formatting instructions carefully.

Format your answer as:
<ANSWER> GROUP NAME: [WORD, WORD, WORD, WORD] </ANSWER>

As a reminder:
{2}- Give your final answer in the format described above (surrounded by <ANSWER> delimiters) without any additional text
- Use the message history to make sure you don't repeat any of your previous guesses

Here are the remaining words:
{0}
'''


INCORRECT_GUESS_PROMPT_ONESHOT = '''
The response from the game was: {1}

Let's continue to solve the puzzle. Again, make you don't repeat any of your previous guesses.

Format your final answers as:
<ANSWER>
GROUP 1 NAME: [WORD, WORD, WORD, WORD]
GROUP 2 NAME: [WORD, WORD, WORD, WORD]
GROUP 3 NAME: [WORD, WORD, WORD, WORD]
GROUP 4 NAME: [WORD, WORD, WORD, WORD]
</ANSWER>

As a reminder:
{2}- Give your final answers in the format described above (surrounded by <ANSWER> delimiters) without any additional text
- Use the message history to make sure you don't repeat any of your previous guesses

The remaining words are:
{0}
'''

INVALID_GUESS_PROMPT_ONESHOT = '''
The response from the game was: {1}

Your answer wasn't formatted correctly. Try again, and follow the formatting instructions carefully.

Format your final answers as:
<ANSWER>
GROUP 1 NAME: [WORD, WORD, WORD, WORD]
GROUP 2 NAME: [WORD, WORD, WORD, WORD]
GROUP 3 NAME: [WORD, WORD, WORD, WORD]
GROUP 4 NAME: [WORD, WORD, WORD, WORD]
</ANSWER>

As a reminder:
{2}- Give your final answers in the format described above (surrounded by <ANSWER> delimiters) without any additional text
- Use the message history to make sure you don't repeat any of your previous guesses

The remaining words are:
{0}
'''