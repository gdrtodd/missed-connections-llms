# The New York Times Connections Puzzle as an LLM Benchmark

Code and data for the IEEE Conference on Games 2024 paper [Missed Connections: Lateral Thinking Puzzles for Large Language Models](https://arxiv.org/pdf/2404.11730)

# Experiments

After installing the requirements with `pip install -r requirements.txt`, the two main experiments in the paper can be run with `python llm_experiment.py` and `python baseline_experiment.py`. Note that for the LLM experiment you will need to set the environment variable `OPENAI_TOKEN` with your API token.

# Data

The data for the first 250 Connections puzzles is available in `data/puzzle_data.json`. To instantiate an instance of the `ConnectionsPuzzle` environment, pass in the corresponding ID of the puzzle from 1 to 250.

