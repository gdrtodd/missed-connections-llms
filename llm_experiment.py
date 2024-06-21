from functools import partial
import json
from itertools import product
import multiprocessing as mp
import os

from tqdm import tqdm

from llm_model import IterativeGPTSolver, OneShotGPTSolver
from utils import solve_puzzle

SOLVER_CHOICES = [IterativeGPTSolver, OneShotGPTSolver]
LLM_CHOICES = ["gpt-4-1106-preview", "gpt-3.5-turbo", ]
CHAIN_OF_THOUGHT = [False, True]
SEEDS = [0, 1, 2]
PUZZLE_IDS = list(range(1, 251))

NUM_GUESSES = 5
INVALID_LIMIT = 5
DATA_DIR = 'data'
SAVE_DIR = 'results'

NUM_PROCS = 8

EXP_VARS = [SOLVER_CHOICES, LLM_CHOICES, CHAIN_OF_THOUGHT, SEEDS, PUZZLE_IDS]

for llm_name in LLM_CHOICES:
    for solver_type in SOLVER_CHOICES:
        for chain_of_thought in CHAIN_OF_THOUGHT:
            description = f"Running {solver_type.__name__}({llm_name}, chain_of_though={chain_of_thought})"

            filename = f"{solver_type.__name__}_{llm_name}_cot-{chain_of_thought}_results.json"

            if os.path.exists(os.path.join(SAVE_DIR, filename)):
                print(f"\nLogs for {description} already exist, checking for missing puzzles / seeds")
                results = json.load(open(os.path.join(SAVE_DIR, filename), "r"))
                seen_puzzles_and_seeds = [(result['puzzle_id'], result['seed']) for result in results]
                
                puzzles_and_seeds = [(puzzle_id, seed) for puzzle_id, seed in product(PUZZLE_IDS, SEEDS) 
                                     if (puzzle_id, seed) not in seen_puzzles_and_seeds]
                total = len(puzzles_and_seeds)

            else:
                print(f"\nLogs for {description} do not exist, running all puzzles / seeds")
                puzzles_and_seeds = list(product(PUZZLE_IDS, SEEDS))
                total = len(PUZZLE_IDS) * len(SEEDS)
                results = []

            if NUM_PROCS == 1:
                for puzzle_id_and_seed in tqdm(puzzles_and_seeds, desc=description, total=total):
                    result = solve_puzzle(puzzle_id_and_seed, solver_type, llm_name, chain_of_thought=chain_of_thought,
                                          num_guesses=NUM_GUESSES, invalid_limit=INVALID_LIMIT)
                    
                    results.append(result)

                    with open(os.path.join(SAVE_DIR, filename), "w") as f:
                        json.dump(results, f)

            else:
                _solve_puzzle = partial(solve_puzzle, solver_type=solver_type, llm_name=llm_name,  chain_of_thought=chain_of_thought,
                                        num_guesses=NUM_GUESSES, invalid_limit=INVALID_LIMIT)
                
                with mp.Pool(NUM_PROCS) as pool:
                    iterator = pool.imap(_solve_puzzle, puzzles_and_seeds)
                    pbar = tqdm(iterator, desc=description, total=total)

                    for result in iterator:
                        results.append(result)
                        pbar.update(1)

                        with open(os.path.join(SAVE_DIR, filename), "w") as f:
                            json.dump(results, f)