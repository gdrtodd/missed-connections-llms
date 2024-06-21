from itertools import combinations
import os
import pickle
import typing

from openai import OpenAI

from puzzle import ConnectionsPuzzle
from llm_model import IterativeGPTSolver, OneShotGPTSolver

def solve_puzzle(puzzle_id_and_seed: typing.Tuple[int, int],
                 solver_type: typing.Union[IterativeGPTSolver, OneShotGPTSolver],
                 llm_name: str,
                 chain_of_thought: bool = False,
                 num_guesses: int = 5,
                 invalid_limit: int = 5):
    
    openai_key = (os.environ.get("OPENAI_TOKEN") or os.environ.get("OPENAI_API_KEY"))
    if openai_key is None:
        raise ValueError("Error: OPENAI_TOKEN/OPENAI_API_KEY environment variable is not set")

    openai_client = OpenAI(api_key=openai_key)
    
    solver = solver_type(openai_client, llm_name, chain_of_thought=chain_of_thought)

    puzzle_id, seed = puzzle_id_and_seed
    all_in_one = isinstance(solver, OneShotGPTSolver)
    puzzle = ConnectionsPuzzle(id=puzzle_id, num_guesses=num_guesses, all_in_one=all_in_one)
    
    solved, invalid_count, step_count, guess_log, messages = solver.solve(puzzle, invalid_limit=invalid_limit, seed=seed)

    solved_overall = solved
    solved_yellow = 'yellow' in puzzle.revealed_colors
    solved_green = 'green' in puzzle.revealed_colors
    solved_blue = 'blue' in puzzle.revealed_colors
    solved_purple = 'purple' in puzzle.revealed_colors
    
    results_dict = {
        'solver': solver_type.__name__,
        'llm_name': llm_name,
        'chain_of_thought': chain_of_thought,
        'puzzle_id': puzzle_id,
        'puzzle_data': puzzle.data,
        'seed': seed,
        'solved_overall': solved_overall,
        'solved_yellow': solved_yellow,
        'solved_green': solved_green,
        'solved_blue': solved_blue,
        'solved_purple': solved_purple,
        'num_steps': step_count,
        'num_invalid': invalid_count,
        'guesses': puzzle.guesses
    }
    
    return results_dict

def enumerate_all_guesses():
    all_guess_idxs = set()

    indices = list(range(16))
    for group_1 in combinations(indices, 4):
        remaining_1 = [idx for idx in indices if idx not in group_1]
        for group_2 in combinations(remaining_1, 4):
            remaining_2 = [idx for idx in remaining_1 if idx not in group_2]
            for group_3 in combinations(remaining_2, 4):
                group_4 = tuple([idx for idx in remaining_2 if idx not in group_3])

                guess_idxs = tuple(sorted((group_1, group_2, group_3, group_4)))

                if guess_idxs not in all_guess_idxs:
                    all_guess_idxs.add(guess_idxs)

    all_guess_idxs = list(all_guess_idxs)

    # Save all_guess_idxs
    with open("./data/all_guess_idxs.pkl", "wb") as f:
        pickle.dump(all_guess_idxs, f)


def get_difficulty_color(color_hex):
    color_dict = {'#df7bea': 'purple',
                  '#fbd400': 'yellow',
                  '#69e352': 'green',
                  '#5492ff': 'blue'}
    
    return color_dict.get(color_hex, None)