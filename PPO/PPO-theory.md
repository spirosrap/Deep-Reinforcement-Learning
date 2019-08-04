# Proximal Policy Optimization Algorithm - Notes from the paper
https://arxiv.org/abs/1707.06347

## Introduction

PPO alternates between sampling data through interaction with the environment, and optimizing a **surrogate objective function** using stochastic gradient ascent.

Whereas standard policy gradient methods perform one gradient update per data sample, we propose a novel objective function that enables multiple epochs of minibatch updates.
