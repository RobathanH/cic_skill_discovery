import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from dm_env import specs
import math
from collections import OrderedDict

import utils

from agent.ddpg import Actor, Critic, Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

"""
Performs Hierarchical RL for downstream fine-tuning of CIC skill-discovery pretraining.
Rather than selecting a single skill vector and fine-tuning its policy, this algorithm
trains a hierarchical policy with discrete actions, and learns a fixed skill vector for
each discrete action.

Use Critic networks in DDPGAgent, predicting value from ground-truth continuous action space
"""

class SkillSelector(nn.Module):
    def __init__(self, skill_vocab_size, skill_dim, obs_type, obs_dim, feature_dim, hidden_dim,
                 skill_vocab_trainable, skill_selector_is_policy):
        super().__init__()
        
        self.skill_selector_is_policy = skill_selector_is_policy
        
        # Parameters representing the skill vector for each action
        skill_vocab = torch.rand(skill_vocab_size, skill_dim)
        self.skill_vocab = nn.Parameter(skill_vocab, requires_grad=skill_vocab_trainable)
        
        if self.skill_selector_is_policy:
            # High-Level skill-selection network (same architecture as ddpg.Agent)
            self.trunk = nn.Sequential(nn.Linear(obs_dim, feature_dim),
                                    nn.LayerNorm(feature_dim), nn.Tanh())
            
            policy_layers = []
            policy_layers += [
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(inplace=True)
            ]
            # add additional hidden layer for pixels
            if obs_type == 'pixels':
                policy_layers += [
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(inplace=True)
                ]
            policy_layers += [nn.Linear(hidden_dim, skill_vocab_size)]

            self.policy = nn.Sequential(*policy_layers)
        
        else:
            # Learn fixed skill selection probabilities that ignore input state
            self.logits = nn.Parameter(torch.zeros(skill_vocab_size), requires_grad=True)

        self.apply(utils.weight_init)
        
    def forward(self, obs, return_logits=False):
        """
        Computes logits for a categorical distribution over the skill vocab
        """
        if self.skill_selector_is_policy:
            h = self.trunk(obs)
            logits = self.policy(h)
        else:
            logits = self.logits[None, :].expand(obs.size(0), -1)
        
        if return_logits:
            return logits
        
        dist = torch.distributions.Categorical(logits=logits)
        samples = dist.sample()
        skills = self.skill_vocab[samples]
        return skills
        
    def compute_skill_vocab_similarity(self):
        """
        Computes current average cosine similarity between skill vectors
        """
        n = self.skill_vocab.size(0)
        
        norm_vocab = F.normalize(self.skill_vocab, dim=1)
        similarities = norm_vocab @ norm_vocab.T
        similarities *= (1 - torch.eye(n, device=device))
        mean_similarity = similarities.sum() / (n * (n - 1))
        return mean_similarity
        
        

class CICHRLAgent:
    def __init__(self, name, skill_vocab_size, skill_dim,
                 skill_entropy_coef, skill_selector_is_policy,
                 skill_vocab_trainable, actor_trainable,
                 obs_type, obs_shape, action_shape,
                 device, lr,
                 feature_dim, hidden_dim,
                 pretrained_feature_dim, pretrained_hidden_dim,
                 critic_target_tau,
                 num_expl_steps, update_every_steps, stddev_schedule,
                 nstep, batch_size, stddev_clip, use_tb, use_wandb):
        self.skill_vocab_size = skill_vocab_size
        self.skill_dim = skill_dim
        self.skill_entropy_coef = skill_entropy_coef
        self.skill_vocab_trainable = skill_vocab_trainable
        self.actor_trainable = actor_trainable
        
        self.obs_type = obs_type
        self.action_dim = action_shape[0]
        self.hidden_dim = hidden_dim
        self.lr = lr
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.use_wandb = use_wandb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip
        self.feature_dim = feature_dim
        self.solved_meta = None

        self.batch_size = batch_size
        
        # Compute obs_dim (potentially after encoder)
        if obs_type == 'pixels':
            self.aug = utils.RandomShiftsAug(pad=4)
            self.encoder = Encoder(obs_shape).to(device)
            self.obs_dim = self.encoder.repr_dim
        else:
            self.aug = nn.Identity()
            self.encoder = nn.Identity()
            self.obs_dim = obs_shape[0]
        
        # High-Level skill vocab trainer and selection policy
        self.skill_selector = SkillSelector(skill_vocab_size, skill_dim, obs_type,
                                            self.obs_dim, feature_dim, hidden_dim,
                                            skill_vocab_trainable, skill_selector_is_policy).to(device)
        
        # Low-Level actor includes skill in input - this is what we load from pretraining
        self.actor = Actor(obs_type, self.obs_dim + skill_dim, self.action_dim,
                           pretrained_feature_dim, pretrained_hidden_dim).to(device)

        # Critic operates on low-level observations and continuous actions, but no skill info
        # TODO: Compare difference with skill input as well
        self.critic = Critic(obs_type, self.obs_dim, self.action_dim,
                             feature_dim, hidden_dim).to(device)
        self.critic_target = Critic(obs_type, self.obs_dim, self.action_dim,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        
        # Optimizers
        self.selector_opt = torch.optim.Adam(self.skill_selector.parameters(), lr=lr)
        
        if obs_type == 'pixels':
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        else:
            self.encoder_opt = None
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        
        if actor_trainable:
            self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        else:
            self.actor_opt = None
        
        self.train()
        
    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)
        self.skill_selector.train(training)
        
    def init_from(self, other):
        utils.hard_update_params(other.encoder, self.encoder)
        utils.hard_update_params(other.actor, self.actor)
            
    # Meta passed from trainer should always be empty
    def get_meta_specs(self):
        return tuple()
    def init_meta(self, time_step=None):
        return OrderedDict()
    def update_meta(self, meta, global_step, time_step, finetune=False):
        return meta
    
    def compute_action_dist(self, obs, skills, step):
        # Append selected skill to input and get action from actor
        
        actor_input = torch.cat([obs, skills], dim=-1)
        stddev = utils.schedule(self.stddev_schedule, step)
        dist = self.actor(actor_input, stddev)
        
        return dist
            
    def act(self, obs, meta, step, eval_mode):
        obs = torch.as_tensor(obs, device=self.device).unsqueeze(0)
        obs = self.encoder(obs)
        
        skills = self.skill_selector(obs)
        dist = self.compute_action_dist(obs, skills, step)
        
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample(clip=None)
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0)
        return action.cpu().numpy()[0]
    
    def update_critic(self, obs, action, reward, discount, next_obs, step):
        metrics = dict()

        with torch.no_grad():
            next_skill = self.skill_selector(obs)
            next_action = self.compute_action_dist(obs, next_skill, step).sample(clip=self.stddev_clip)
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)
            target_Q = reward + (discount * target_V)

        Q1, Q2 = self.critic(obs, action)
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q)

        if self.use_tb or self.use_wandb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize critic
        if self.encoder_opt is not None:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.encoder_opt is not None:
            self.encoder_opt.step()
        return metrics
    
    def reset_critic(self):
        # reset critic parameters
        self.critic.reset_params()
        # reset critic target
        self.critic_target.load_state_dict(self.critic.state_dict())
        
    def update_actor(self, obs, step):
        metrics = dict()
        
        selection_logits = self.skill_selector(obs, return_logits=True)
        skill_probs = torch.softmax(selection_logits, dim=-1)

        # Compute action and action-value for each possible skill on each observation
        obs = obs[:, None, :].expand(-1, self.skill_vocab_size, -1) # (bsz, skill_vocab, obs_dim)
        skills = self.skill_selector.skill_vocab[None, :, :].expand(obs.size(0), -1, -1) # (bsz, skill_vocab, skill_dim)
        obs = obs.flatten(end_dim=1)
        skills = skills.flatten(end_dim=1)
        dist = self.compute_action_dist(obs, skills, step)
        action = dist.sample(clip=self.stddev_clip)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)
        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)
        Q = Q.reshape(-1, self.skill_vocab_size)

        # Take expectation of value over skills
        Q = (skill_probs * Q).sum(-1)

        actor_loss = -Q.mean()
        
        # Add normalized skill entropy term
        skill_entropy = torch.distributions.Categorical(logits=selection_logits).entropy().mean() / math.log(self.skill_vocab_size)
        actor_loss -= self.skill_entropy_coef * skill_entropy

        # optimize skill selector
        self.selector_opt.zero_grad(set_to_none=True)
        if self.actor_trainable:
            self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.selector_opt.step()
        if self.actor_trainable:
            self.actor_opt.step()

        if self.use_tb or self.use_wandb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()
            metrics["skill_selector_entropy"] = skill_entropy.item()
            metrics["skill_vocab_similarity"] = self.skill_selector.compute_skill_vocab_similarity().item()

        return metrics
    
    def aug_and_encode(self, obs):
        obs = self.aug(obs)
        return self.encoder(obs)
    
    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            return metrics
        
        batch = next(replay_iter)
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device)
        
        # augment and encode
        obs = self.aug_and_encode(obs)
        with torch.no_grad():
            next_obs = self.aug_and_encode(next_obs)
            
        if self.use_tb or self.use_wandb:
            metrics['batch_reward'] = reward.mean().item()
            
        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
