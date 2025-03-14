"""Sequential implementation of Recurrent Neural Network Language Model."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
import os
import re
import shutil
from typing import Any, Collection, Dict, Iterable, Optional, Pattern, Sequence, Set, Tuple, cast

from espnet.nets.lm_interface import LMInterface
from pygtrie import CharTrie

# def _get_empty_lm_state() -> "kenlm.State":
#     """Get unintialized kenlm state."""
#     try:
#         kenlm_state = kenlm.State()
#     except ImportError:
#         raise ValueError("To use a language model, you need to install kenlm.")
#     return kenlm_state



# def _prepare_unigram_set(unigrams: Collection[str], kenlm_model: "kenlm.Model") -> Set[str]:
#     """Filter unigrams down to vocabulary that exists in kenlm_model."""
#     if len(unigrams) < 1000:
#         logger.warning(
#             "Only %s unigrams passed as vocabulary. Is this small or artificial data?",
#             len(unigrams),
#         )
#     unigram_set = set(unigrams)
#     unigram_set = set([t for t in unigram_set if t in kenlm_model])
#     retained_fraction = 1.0 if len(unigrams) == 0 else len(unigram_set) / len(unigrams)
#     if retained_fraction < 0.1:
#         logger.warning(
#             "Only %s%% of unigrams in vocabulary found in kenlm model-- this might mean that your "
#             "vocabulary and language model are incompatible. Is this intentional?",
#             round(retained_fraction * 100, 1),
#         )
#     return unigram_set



# class KenLM(LMInterface):
#     """KenlLM.

#     See also:
#         https://github.com/kensho-technologies/pyctcdecode/blob/main/pyctcdecode/language_model.py

#     """

#     def __init__(self, n_vocab, args):
#         """Initialize class.

#         Args:
#             n_vocab (int): The size of the vocabulary
#             args (argparse.Namespace): configurations. see py:method:`add_arguments`

#         """
#         torch.nn.Module.__init__(self)
#         self._kenlm_model = kenlm_model
#         unigram_set = _prepare_unigram_set(unigrams, self._kenlm_model)
#         char_trie = CharTrie.fromkeys(unigram_set)
#         self._unigram_set = unigram_set
#         self._char_trie = char_trie
#         self.alpha = alpha
#         self.beta = beta
#         self.unk_score_offset = unk_score_offset
#         self.score_boundary = score_boundary

#     def score(self, state, word):
#         """Score new token.

#         Args:
#             y (torch.Tensor): 1D torch.int64 prefix tokens. -> X
#             state: Scorer state for prefix tokens
#             x (torch.Tensor): 2D encoder feature that generates ys. -> word

#         Returns:
#             tuple[torch.Tensor, Any]: Tuple of
#                 torch.float32 scores for next token (n_vocab)
#                 and next state for ys
#         해당 입력 x 에 대한 logp와 새로운 state를 출력해야함. 요거만 맞추면 됨 !!
#         """
#         # self, prev_state: AbstractLMState, word: str, is_last_word: bool = False
#         # ) -> Tuple[float, KenlmState]:
#         """Score word conditional on start state."""
#         if not isinstance(state, KenlmState):
#             raise AssertionError(
#                 f"Wrong input state type found. Expected KenlmState, got {type(prev_state)}"
#             )
#         end_state = _get_empty_lm_state()
#         lm_score = self._kenlm_model.BaseScore(prev_state.state, word, end_state)
#         # override UNK prob. use unigram set if we have because it's faster
#         if (
#             len(self._unigram_set) > 0
#             and word not in self._unigram_set
#             or word not in self._kenlm_model
#         ):
#             lm_score += self.unk_score_offset
#         # add end of sentence context if needed
#         if is_last_word:
#             # note that we want to return the unmodified end_state to keep extension capabilities
#             lm_score = lm_score + self._get_raw_end_score(end_state)
#         lm_score = self.alpha * lm_score * LOG_BASE_CHANGE_FACTOR + self.beta
#         return lm_score, KenlmState(end_state)
    
    
"""    
ChatGPT 버전
"""    
import kenlm
import torch
import torch.nn as nn
import torch.nn.functional as F

from espnet.nets.lm_interface import LMInterface


class KenLMLanguageModel(LMInterface, nn.Module):
    """KenLM 기반 언어 모델 클래스."""

    @staticmethod
    def add_arguments(parser):
        """명령줄 인자 추가."""
        parser.add_argument("--kenlm_model", type=str, required=True, help="KenLM 모델 파일 경로")
        return parser

    def __init__(self, kenlm_model_path, args):
        """KenLM 모델을 로드하고 초기화.

        Args:
            kenlm_model_path (str): 학습된 KenLM 바이너리 모델 파일 경로.
        """
        super().__init__()
        with open(kenlm_model_path, "r", encoding="utf-8") as f:
            self.config = json.load(f)
        self.model = kenlm.Model(self.config["binary_file"])

    def forward(self, x, t):
        """KenLM을 사용하여 손실(loss) 계산.

        Args:
            x (list of str): 입력 텍스트 (batch 크기만큼 리스트)
            t (list of str): 타겟 텍스트 (batch 크기만큼 리스트)

        Returns:
            tuple(torch.Tensor, torch.Tensor, torch.Tensor):
                - 평균 log-likelihood
                - 전체 log-likelihood 합산
                - 샘플 개수
        """
        batch_size = len(x)
        log_probs = []
        num_tokens = []

        for i in range(batch_size):
            sentence = " ".join(x[i])  # 입력을 하나의 문장으로 변환
            log_prob = self.kenlm_model.score(sentence)  # KenLM에서 log-prob 계산
            log_probs.append(log_prob)
            num_tokens.append(len(x[i]))  # 문장의 단어 수

        log_probs = torch.tensor(log_probs, dtype=torch.float32)
        num_tokens = torch.tensor(num_tokens, dtype=torch.float32)

        avg_log_prob = log_probs.mean()  # 평균 log-likelihood
        total_log_prob = log_probs.sum()  # 전체 log-likelihood
        total_count = num_tokens.sum()  # 전체 단어 개수

        return avg_log_prob, total_log_prob, total_count

    def score(self, y, state, x):
        """KenLM을 사용하여 새 토큰 예측 확률을 반환.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: 이전 상태 (KenLM에서는 따로 사용하지 않음)
            x (torch.Tensor): 2D encoder feature (사용되지 않음)

        Returns:
            tuple[torch.Tensor, None]:
                - torch.float32 확률 (n_vocab 크기)
                - 다음 state (None)
        """
        sentence = " ".join(y.tolist())  # 토큰 리스트를 문자열로 변환
        logp = self.kenlm_model.score(sentence)  # 문장 확률 계산
        return torch.tensor(logp, dtype=torch.float32), None

    def get_perplexity(self, sentence):
        """Perplexity(혼란도)를 계산.

        Args:
            sentence (str): 평가할 문장.

        Returns:
            float: Perplexity 값.
        """
        words = sentence.split()
        logp = self.kenlm_model.score(sentence)
        return torch.exp(-torch.tensor(logp / len(words)))  # Perplexity = exp(-log prob / num tokens)
