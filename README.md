# Skill-Tuning in Pretrained Skill-Conditioned Locomotion Policies

Research project with Stanford's CS 224R Deep Reinforcement Learning course.

This project investigates alternate approaches to fine-tuning a frozen locomotion policy pretrained with unsupervised skill discovery in the Mujoco simulator. In particular, I use Contrastive Intrinsic Control (CIC) as the pretraining algorithm, which produces a policy conditioned on a continuous latent skill representation. Unlike CIC, which performs task-specific finetuning by freezing the skill vector and training the full network, this alternative focuses on tuning the skill vectors directly, similar to Context Optimization methods in language models. The primary method (`agent/cic_hrl.py`) optimizes a vocabulary of skill vectors alongside a hierarchical skill-selection policy, partially reducing a continuous action space into a discrete action space.

For more details see the [poster](skill_tuning_poster.pdf) or the [report](skill_tuning_report.pdf).

To install and run, follow the instructions laid out in the [original readme](original_readme.md). Additionally, [install.md](install.md) lists the steps I found to work when installing.
