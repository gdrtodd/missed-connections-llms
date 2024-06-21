from functools import partial
import json
from itertools import product
import multiprocessing as mp
import os
import typing

from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from baselines import SentenceTransformerBaseline, ClustersBaseline
from puzzle import ConnectionsPuzzle

def solve_puzzle(puzzle_id: int, solver_type: typing.Union[SentenceTransformerBaseline, ClustersBaseline],
                 model_name: str = "all-MiniLM-L6-v2", num_guesses: int = 5):
    
    solver = solver_type(model_name=model_name)
    all_in_one = isinstance(solver, ClustersBaseline)
    puzzle = ConnectionsPuzzle(id=puzzle_id, num_guesses=num_guesses, all_in_one=all_in_one)

    observation, done, reward = puzzle.reset()
    solver.reset()

    category_solved_at = {}

    with tqdm(range(num_guesses), desc=f"Solving puzzle {puzzle_id}", leave=False) as pbar:
        while not done:
            guess = solver.get_action(observation)
            observation, done, reward = puzzle.step(guess)

            for color in puzzle.revealed_colors:
                if color not in category_solved_at:
                    category_solved_at[color] = num_guesses - puzzle.guesses_remaining

            pbar.update(1)


    solved_overall = (reward == 1)

    solved_yellow = 'yellow' in puzzle.revealed_colors
    yellow_solved_at = category_solved_at['yellow'] if solved_yellow else None

    solved_green = 'green' in puzzle.revealed_colors
    green_solved_at = category_solved_at['green'] if solved_green else None

    solved_blue = 'blue' in puzzle.revealed_colors
    blue_solved_at = category_solved_at['blue'] if solved_blue else None

    solved_purple = 'purple' in puzzle.revealed_colors
    purple_solved_at = category_solved_at['purple'] if solved_purple else None
    
    results_dict = {
        'puzzle_id': puzzle_id,
        'puzzle_data': puzzle.data,
        'solved_overall': solved_overall,

        'solved_yellow': solved_yellow,
        'yellow_solved_at': yellow_solved_at,

        'solved_green': solved_green,
        'green_solved_at': green_solved_at,

        'solved_blue': solved_blue,
        'blue_solved_at': blue_solved_at,
        
        'solved_purple': solved_purple,
        'purple_solved_at': purple_solved_at,
    }

    return results_dict

SOLVER_CHOICES = [SentenceTransformerBaseline]
MODEL_NAMES = ["all-MiniLM-L6-v2", 'bert-base-nli-mean-tokens', 'all-roberta-large-v1', 'all-mpnet-base-v2']
PUZZLE_IDS = list(range(1, 251))
NUM_GUESSES = 500

DATA_DIR = 'data'
SAVE_DIR = 'results'

NUM_PROCS = 1

for solver_type in SOLVER_CHOICES:
    for model_name in MODEL_NAMES:
        results = []
        filename = f"{solver_type.__name__}_model-{model_name}_results.json"

        if os.path.exists(os.path.join(SAVE_DIR, filename)):
            print(f"\nLogs for {solver_type.__name__} already exist, checking for missing puzzles...")
            results = json.load(open(os.path.join(SAVE_DIR, filename), "r"))
            seen_ids = [result['puzzle_id'] for result in results]
            puzzle_ids = [puzzle_id for puzzle_id in PUZZLE_IDS if puzzle_id not in seen_ids]
            total = len(puzzle_ids)

        else:
            print(f"\nLogs for {solver_type.__name__} do not exist, running all puzzles / seeds")
            puzzle_ids = PUZZLE_IDS
            total = len(PUZZLE_IDS)
            results = []

        if NUM_PROCS == 1:
            for puzzle_id in tqdm(puzzle_ids, desc=f"Running {solver_type.__name__}-{model_name}", total=total):
                result = solve_puzzle(puzzle_id, solver_type, model_name, num_guesses=NUM_GUESSES)
                results.append(result)

                with open(os.path.join(SAVE_DIR, filename), "w") as f:
                    json.dump(results, f)
        
        else:
            _solve_puzzle = partial(solve_puzzle, solver_type=solver_type, model_name=model_name, num_guesses=NUM_GUESSES)
            with mp.Pool(NUM_PROCS) as pool:
                iterator = pool.imap(_solve_puzzle, puzzle_ids)
                pbar = tqdm(iterator, desc=f"Running {solver_type.__name__}", total=total)
                
                for result in pbar:
                    results.append(result)
                    pbar.update(1)

                with open(os.path.join(SAVE_DIR, filename), "w") as f:
                    json.dump(results, f)