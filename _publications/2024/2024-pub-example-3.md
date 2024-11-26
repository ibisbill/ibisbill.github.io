---
title:          "World Models with Hints of Large Language Models for Goal Achieving"
date:           2024-05-12 00:01:00 +0800
selected:       true
pub:            "NAACL"
# pub_pre:        "Submitted to "
pub_post:       'Under review.'
# pub_last:       ' <span class="badge badge-pill badge-publication badge-success">Spotlight</span>'
pub_date:       "2024"

abstract: >-
Reinforcement learning struggles in the face of long-horizon tasks and sparse goals due to the difficulty in manual reward specification. While existing methods address this by adding intrinsic rewards, they may fail to provide meaningful guidance in long-horizon decision-making tasks with large state and action spaces,
lacking purposeful exploration. Inspired by human cognition, we propose a new multi-modal model-based RL approach named Dreaming with Large Language Models (DLLM). DLLM integrates the proposed hinting subgoals from the LLMs into the model rollouts to encourage goal discovery and reaching in challenging tasks. By assigning higher intrinsic rewards to samples that align with the hints outlined by the language model during model rollouts, DLLM guides the agent toward meaningful and efficient exploration. Extensive experiments demonstrate
that the DLLM outperforms recent methods in various challenging, sparse-reward environments such as HomeGrid, Crafter, and Minecraft by 27.7%, 21.1%, and 9.9%, respectively.
cover:          /assets/images/covers/cover3.jpg
authors:
  - Zeyuan Liu*
  - Maggie Ziyu Huan*
  - Xiyao Wang
  - Jiafei Lyu
  - Jian Tao
  - Xiu Li
  - Furong Huang
  - Huazhe Xu
links:
  # Code: https://github.com/luost26/academic-homepage
  Paper: https://arxiv.org/pdf/2406.07381 
---
