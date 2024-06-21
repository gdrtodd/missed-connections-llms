from itertools import combinations, product
import math
import multiprocessing as mp
import os
import pickle
import typing

os.environ['TOKENIZERS_PARALLELISM'] = 'false'

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from sentence_transformers import SentenceTransformer, util
from sklearn.cluster import KMeans
from tqdm import tqdm

class SentenceTransformerBaseline():
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 aggregation_fn: typing.Callable = np.mean):
        
        self.model = SentenceTransformer(model_name)
        self.aggregation_fn = aggregation_fn

        self.guesses = []
        self.embeddings = None
        self.cosine_scores = None
        self.words_to_idx = None
        self.initial_words = None

    def reset(self):
        '''
        Reset for a new puzzle
        '''
        self.guesses = []
        self.embeddings = None
        self.cosine_scores = None
        self.words_to_idx = None
        self.initial_words = None

        self.available_indices_to_ranked_guesses = {}

    def get_action(self, observation: dict) -> typing.List[str]:
        '''
        Determine an action for the current observation
        '''

        words = observation["words"]
        
        # Cache embeddings and cosine similarties
        if self.embeddings is None:

            self.embeddings = self.model.encode(words, convert_to_tensor=True)
            self.cosine_scores = util.cos_sim(self.embeddings, self.embeddings).cpu()
            self.words_to_idx = {word: idx for idx, word in enumerate(words)}
            self.initial_words = words[:]

        indices = tuple(sorted([self.words_to_idx[word] for word in words]))

        if indices in self.available_indices_to_ranked_guesses:
            ranked_guesses = self.available_indices_to_ranked_guesses[indices]
        else:
            def score_group(group):
                idx_group = [self.words_to_idx[word] for word in group]
                scores = []
                for i, j in product(idx_group, repeat=2):
                    scores.append(self.cosine_scores[i, j])

                return self.aggregation_fn(scores)

            groups = []
            for idx_group in tqdm(combinations(indices, 4), desc="Generating guesses", total=math.comb(len(indices), 4), leave=False):
                group = list(sorted([self.initial_words[idx] for idx in idx_group]))
                groups.append(group)
            
            ranked_guesses = sorted(groups, key=score_group, reverse=True)
            self.available_indices_to_ranked_guesses[indices] = ranked_guesses

        guess = ranked_guesses.pop(0)
        while guess in self.guesses:
            guess = ranked_guesses.pop(0)

        self.guesses.append(guess)

        return guess
    
    def get_action_old(self, observation: dict) -> typing.List[str]:
        '''
        Determine an action for the current observation
        '''

        words = observation["words"]
        
        # Cache embeddings and cosine similarties
        if self.embeddings is None:

            self.embeddings = self.model.encode(words, convert_to_tensor=True)
            self.cosine_scores = util.cos_sim(self.embeddings, self.embeddings).cpu()
            self.words_to_idx = {word: idx for idx, word in enumerate(words)}
            self.initial_words = words[:]

        indices = [self.words_to_idx[word] for word in words]

        # Find groups of four items that have the 'best' cosine similarity
        # according to the aggregation function
        best_group = None
        best_score = -np.inf
        for idx_group in combinations(indices, 4):
            group = list(sorted([self.initial_words[idx] for idx in idx_group]))
            if group in self.guesses:
                continue

            scores = []
            for i, j in product(idx_group, repeat=2):
                scores.append(self.cosine_scores[i, j])

            score = self.aggregation_fn(scores)
            if score > best_score:
                best_score = score
                best_group = group

        self.guesses.append(best_group)

        return best_group

class ClustersBaseline():
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 aggregation_fn: typing.Callable = np.mean):
        
        self.model = SentenceTransformer(model_name)
        self.aggregation_fn = aggregation_fn

        self.guesses = []
        self.embeddings = None
        self.cosine_scores = None
        self.words_to_idx = None
        self.initial_words = None

        self.all_group_idxs = pickle.load(open("./data/all_guess_idxs.pkl", "rb"))
        self.group_to_score = {}

    def reset(self):
        '''
        Reset for a new puzzle
        '''
        self.guesses = []
        self.embeddings = None
        self.cosine_scores = None
        self.words_to_idx = None
        self.initial_words = None

        self.group_to_score = {}

    def _score_group(self, group: typing.Tuple[int]) -> float:
        '''
        Determine the aggregate cosine similarity between all pairs of words in a group
        '''

        if group not in self.group_to_score:

            scores = []
            for i, j in product(group, repeat=2):
                scores.append(self.cosine_scores[i, j])

            score = self.aggregation_fn(scores)

            self.group_to_score[group] = score

        return self.group_to_score[group]


    def _score_guess(self, guess_idxs: typing.Tuple[typing.Tuple[int]]) -> typing.Tuple[float, typing.List[typing.List[str]]]:
        
        score = sum([self._score_group(group) for group in guess_idxs])
        guess = [[self.words[idx] for idx in group] for group in guess_idxs]

        return score, guess

    def get_action(self, observation: dict) -> typing.List[str]:
        '''
        Determine an action for the current observation
        '''

        self.words = observation["words"]
        
        # Cache embeddings and cosine similarties
        if self.embeddings is None:

            self.embeddings = self.model.encode(self.words, convert_to_tensor=True)
            self.cosine_scores = util.cos_sim(self.embeddings, self.embeddings)
            self.guesses_by_score = []

            for guess_idxs in tqdm(self.all_group_idxs, desc="Generating guesses", total=len(self.all_group_idxs)):
                score, guess = self._score_guess(guess_idxs)
                self.guesses_by_score.append((score, guess))

            self.guesses_by_score = sorted(self.guesses_by_score, key=lambda x: x[0], reverse=True)

        for score, guess in self.guesses_by_score:
            if guess not in self.guesses:
                self.guesses.append(guess)
                return guess
            
        raise ValueError("No more guesses available")


class KMeansBaseline():
    def __init__(self,
                 model_name: str = "all-MiniLM-L6-v2",
                 seed: int = 0):
        
        self.model = SentenceTransformer(model_name)
        self.seed = seed

    def reset(self):
        '''
        Reset for a new puzzle
        '''
        pass

    def get_action(self, observation: dict) -> typing.List[typing.List[str]]:
        '''
        Determine an action for the current observation
        '''
        words = observation["words"]
        embeddings = self.model.encode(words, convert_to_numpy=True)
        word_to_embedding = {word: embedding for word, embedding in zip(words, embeddings)}

        kmeans = KMeans(n_clusters=4, n_init="auto").fit(embeddings)
        # labels = kmeans.labels_

        # Code from https://stackoverflow.com/questions/5452576/k-means-algorithm-variation-with-equal-cluster-size
        # for ensuring equal cluster size
        centers = kmeans.cluster_centers_
        centers = centers.reshape(-1, 1, embeddings.shape[-1]).repeat(4, 1).reshape(-1, embeddings.shape[-1])
        distance_matrix = cdist(embeddings, centers)
        labels = linear_sum_assignment(distance_matrix)[1] // 4

        action = []
        for label in range(4):
            action.append([word for word, word_label in zip(words, labels) if word_label == label])

        return action