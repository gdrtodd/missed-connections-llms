import os
import re
import typing

import backoff
import openai
from openai import OpenAI

from puzzle import ConnectionsPuzzle, PuzzleReponse
from prompts import *

class GPTSolver():
    def __init__(self,
                 openai_client: OpenAI,
                 openai_model_str: str,
                 max_openai_tokens: int = 1024,
                 openai_temperature: float = 0.0):

        # Instantiate the client
        self.client = openai_client

        self.openai_model_str = openai_model_str
        self.max_openai_tokens = max_openai_tokens
        self.openai_temperature = openai_temperature

    @backoff.on_exception(backoff.expo, openai.RateLimitError)
    def _query_openai(self, messages):
        '''
        Query the specified openai model with the given prompt, and return the response. Assumes
        that the API key has already been set. Retries with exponentially-increasing delays in
        case of rate limit errors
        '''


        response = self.client.chat.completions.with_raw_response.create(
                    model=self.openai_model_str,
                    max_tokens=self.max_openai_tokens,
                    temperature=self.openai_temperature,
                    messages=messages,
                    seed=self.seed
            )
        completion = response.parse()

        reponse_content = completion.choices[0].message.content

        return reponse_content
    
    def reset(self):
        '''
        Reset the solver for a new puzzle
        '''
        pass
    
class IterativeGPTSolver(GPTSolver):
    '''
    LLM-based solver for the Connections puzzle using an iterative approach, providing all previous message history along the way
    '''
    def __init__(self,
                 openai_client: OpenAI,
                 openai_model_str: str,
                 max_openai_tokens: int = 1024,
                 openai_temperature: float = 0.0,
                 use_system_prompt: bool = False,
                 chain_of_thought: bool = False):

        super().__init__(openai_client, openai_model_str, max_openai_tokens, openai_temperature)

        # Matches to text inbetween <ANSWER> delimiters
        self.answer_regex = r"(?<=<ANSWER>)([\S\s]*?)(?=</ANSWER>)"
    
        self.prompt_mapping = {
             "INITIAL": INITIAL_PROMPT_ITERATIVE,
             "CORRECT": CORRECT_GUESS_PROMPT_ITERATIVE,
             "NEARLY_CORRECT": NEARLY_CORRECT_GUESS_PROMPT_ITERATIVE,
             "INCORRECT": INCORRECT_GUESS_PROMPT_ITERATIVE,
             "INVALID": INVALID_GUESS_PROMPT_ITERATIVE
        }

        self.use_system_prompt = use_system_prompt
        self.cot_injection = COT_PROMPT_ITERATIVE if chain_of_thought else ""
    
    def solve(self, puzzle: ConnectionsPuzzle, invalid_limit: int = 5, seed: int = 0):
        '''
        Attempt to solve the provided puzzle by querying the language model
        '''

        # Set the seed
        self.seed = seed
        
        invalid_count = 0
        step_count = 0
        guess_log = []

        # Messages contains the list of interleaved prompts and responses from the LLM
        llm_messages = []

        # Reset the puzzle to obtain initial observation
        observation, done, reward = puzzle.reset()
        game_message = observation['message']

        # Add the initial prompt to messages
        if self.use_system_prompt:
            llm_messages.append({"role": "system", "content": SYSTEM_PROMPT})
        llm_messages.append({"role": "user", "content": self.prompt_mapping["INITIAL"].format(puzzle.words, self.cot_injection)})

        done = False
        reward = 0

        # continue until we solve the puzzle, run out of guesses, or have too many invalid attempts (leads to it getting stuck in a loop)        
        while not done and invalid_count < invalid_limit:

            # Query the LLM with the current message history and record its response
            llm_response = self._query_openai(llm_messages)

            # Attempt to parse the guess from the LLM response
            answer_match = re.findall(self.answer_regex, llm_response)

            if answer_match:
                words_match = re.findall(r"(?<=\[)(.*)(?=\])", answer_match[0])
                guess = words_match[0].split(",") if words_match else []

                observation, done, reward = puzzle.step(guess)

                game_response = observation["response"]
                game_message = observation['message']

                if game_response == PuzzleReponse.CORRECT:
                    next_prompt = self.prompt_mapping["CORRECT"]
                    guess_log.append(str(guess))
                
                elif game_response == PuzzleReponse.NEARLY_CORRECT:
                    next_prompt = self.prompt_mapping["NEARLY_CORRECT"]
                    guess_log.append(str(guess))

                elif game_response == PuzzleReponse.INCORRECT:
                    next_prompt = self.prompt_mapping["INCORRECT"]
                    guess_log.append(str(guess))

                elif game_response == PuzzleReponse.INVALID:
                    next_prompt = self.prompt_mapping["INVALID"]
                    invalid_count += 1
                    guess_log.append("Malformed guess")
                
            else:
                game_message = ""
                next_prompt = self.prompt_mapping["INVALID"]
                invalid_count += 1
                guess_log.append("Failed regex parse")

            # Inject info into new prompt and add to message history
            llm_messages.append({"role": "user", "content": next_prompt.format(puzzle.words, game_message, self.cot_injection)})
            step_count += 1

        solved = (reward == 1)
        return solved, invalid_count, step_count, guess_log, llm_messages


class OneShotGPTSolver(GPTSolver):
    def __init__(self,
                 openai_client: OpenAI,
                 openai_model_str: str,
                 max_openai_tokens: int = 1024,
                 openai_temperature: float = 0.0,
                 use_system_prompt: bool = False,
                 chain_of_thought: bool = False):


        super().__init__(openai_client, openai_model_str, max_openai_tokens, openai_temperature)

        self.prompt_mapping = {
             "INITIAL": INITIAL_PROMPT_ONESHOT,
             "INCORRECT": INCORRECT_GUESS_PROMPT_ONESHOT,
             "INVALID": INVALID_GUESS_PROMPT_ONESHOT
        }

        self.use_system_prompt = use_system_prompt
        self.cot_injection = COT_PROMPT_ONESHOT if chain_of_thought else ""

        # Matches to text inbetween <ANSWER> delimiters
        self.answer_regex = r"(?<=<ANSWER>)([\S\s]*?)(?=</ANSWER>)"

    def solve(self, puzzle: ConnectionsPuzzle, invalid_limit: int = 5, seed: int = 0):
        '''
        Attempt to solve the provided puzzle by querying the language model
        '''

        # Set the seed
        self.seed = seed

        assert puzzle.all_in_one, "Pure one-shot solver only works for all-in-one puzzles"

        invalid_count = 0
        step_count = 0
        guess_log = []

        # Messages contains the list of interleaved prompts and responses from the LLM
        llm_messages = []

        # Reset the puzzle to obtain initial observation
        observation, done, reward = puzzle.reset()
        game_message = observation['message']

        done = False
        reward = 0

        # Add the initial prompt to messages
        if self.use_system_prompt:
            llm_messages.append({"role": "system", "content": SYSTEM_PROMPT})
        llm_messages.append({"role": "user", "content": self.prompt_mapping["INITIAL"].format(puzzle.words, self.cot_injection)})

        done = False
        reward = 0

        while invalid_count < invalid_limit and not done:

            # Query the LLM with the current message history and record its response
            llm_response = self._query_openai(llm_messages)
            llm_messages.append({"role": "assistant", "content": llm_response})

            # Attempt to parse the guess from the LLM response
            # Attempt to parse the guess from the LLM response
            answer_match = re.findall(self.answer_regex, llm_response)

            if answer_match:
                words_match = re.findall(r"(?<=\[)(.*)(?=\])", answer_match[0])
                guess = [match.split(",") for match in words_match]

                observation, done, reward = puzzle.step(guess)
                game_response = observation["response"]
                game_message = observation['message']

                if game_response == PuzzleReponse.CORRECT:
                    next_prompt = ""
                    guess_log.append(str(guess))

                elif game_response == PuzzleReponse.INCORRECT:
                    next_prompt = self.prompt_mapping["INCORRECT"]
                    guess_log.append(str(guess))

                elif game_response == PuzzleReponse.INVALID:
                    next_prompt = self.prompt_mapping["INVALID"]
                    invalid_count += 1
                    guess_log.append("Malformed guess")

                
            else:
                # Did not parse 4 groups from llm response
                invalid_count += 1
                next_prompt = self.prompt_mapping["INVALID"]
                guess_log.append("Failed regex parse")

            # Inject info into new prompt and add to message history
            llm_messages.append({"role": "user", "content": next_prompt.format(puzzle.words, game_message, self.cot_injection)})
            step_count += 1

        solved = (reward == 1)
        return solved, invalid_count, step_count, guess_log, llm_messages