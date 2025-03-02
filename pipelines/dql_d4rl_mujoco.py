import os
from copy import deepcopy
import logging
import d4rl
import gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
import torch.autograd as autograd
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from cleandiffuser.dataset.d4rl_mujoco_dataset import D4RLMuJoCoTDDataset
from cleandiffuser.dataset.dataset_utils import loop_dataloader
from cleandiffuser.diffusion import DiscreteDiffusionSDE
from cleandiffuser.nn_condition import IdentityCondition
from cleandiffuser.nn_diffusion import DQLMlp
from cleandiffuser.utils import report_parameters, DQLCritic, FreezeModules
from cleandiffuser.STAC.actors.kernels import RBF
from utils import set_seed
import warnings

warnings.filterwarnings("ignore")

def svgd_update(a_0, s, itr_num, svgd_step, num_particles, act_dim, critic, kernel, device):
    a = a_0 # always initiate the particles with the behavioral actions
    KL = torch.zeros(num_particles,num_particles).to(device) 
    identity = torch.eye(num_particles).to(device)
    for l in range(itr_num):
        q_1, q_2 = critic(s,a)
        q = torch.min(q_1,q_2)
        # print(q_1.size(), q_2.size(),s.size(),a.size()) # [256,1],[256,1],[256,17],[256,6]
        score_func = autograd.grad(q.sum(), a, retain_graph=True, create_graph=True)[0]
        a = a.reshape(-1, num_particles, act_dim)
        # print(a.size()) # [16,16,6]
        score_func = score_func.reshape(a.size())
        # print(score_func.mean())
        K_value, K_dist_sq, K_gamma, K_grad = kernel(a, a) # [16,16,16],[16,16,16],[16,1,1],[16,16,16,6]
        # print(K_value.size(),K_dist_sq.size(),K_gamma.size(),K_grad.size())
        h = (K_value.matmul(score_func) + K_grad.sum(1)) / num_particles
        # print(h.size()) #[16,16,6]  
        a, h = a.reshape(-1,act_dim), h.reshape(-1,act_dim)
        # compute the KL divergence
        score_func = score_func.reshape(num_particles, act_dim, num_particles).unsqueeze(1) # [16,1,6,16]
        term1 = (K_grad.matmul(score_func)).sum(-1).sum(-1)/(num_particles-1) # [16,16]
        term2 = K_value.matmul(act_dim * K_gamma - K_dist_sq * K_gamma.pow(2)).sum(-1) / (num_particles-1) # [16,16]
        term1, term2 = term1.to(device), term2.to(device)
        # print(term1.mean(),term2.mean())
        # print(term1.size(),term2.size())
        KL = KL + svgd_step * (term1 + term2)
        a = a + svgd_step * h # update the particles
    KL = KL.reshape(num_particles*num_particles,1)
    return a, KL # should be of size [256,6] and [256,1]


@hydra.main(config_path="../configs/dql/mujoco", config_name="mujoco", version_base=None)
def pipeline(args):

    set_seed(args.seed)

    save_path = f'results/{args.pipeline_name}/{args.task.env_name}/'
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # ---------------------- Create Dataset ----------------------
    env = gym.make(args.task.env_name)
    dataset = D4RLMuJoCoTDDataset(d4rl.qlearning_dataset(env), args.normalize_reward)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    obs_dim, act_dim = dataset.o_dim, dataset.a_dim

    # --------------- Network Architecture -----------------
    nn_diffusion = DQLMlp(obs_dim, act_dim, emb_dim=64, timestep_emb_type="positional").to(args.device)
    nn_condition = IdentityCondition(dropout=0.0).to(args.device)

    logger = logging.getLogger('my_logger')
    logger.setLevel(logging.INFO)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(os.path.join(script_dir,'logs'), exist_ok=True)
    log_path = os.path.join(script_dir, 'logs', 'pretrain-itr{}-alpha{}'.format(args.itr_num,args.alpha))
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    print(f"======================= Parameter Report of Diffusion Model =======================")
    report_parameters(nn_diffusion)
    print(f"==============================================================================")

    # ---------------------- Actor ----------------------
    actor = DiscreteDiffusionSDE(
        nn_diffusion, nn_condition, predict_noise=args.predict_noise, optim_params={"lr": args.actor_learning_rate},
        x_max=+1. * torch.ones((1, act_dim), device=args.device),
        x_min=-1. * torch.ones((1, act_dim), device=args.device),
        diffusion_steps=args.diffusion_steps, ema_rate=args.ema_rate, device=args.device)
    # ------------------ Critic ---------------------
    critic = DQLCritic(obs_dim, act_dim, hidden_dim=args.hidden_dim).to(args.device)
    critic_target = deepcopy(critic).requires_grad_(False).eval()
    critic_optim = torch.optim.Adam(critic.parameters(), lr=args.critic_learning_rate)

    # ---------------------- Pretraining a diffusion model sampler ----------------------
    actor_lr_scheduler = CosineAnnealingLR(actor.optimizer, T_max=args.gradient_steps)
    critic_lr_scheduler = CosineAnnealingLR(critic_optim, T_max=args.gradient_steps)
    actor.train()
    critic.train()
    pretrain_gradient_step = 0
    logger.info("Starting to pretrain the diffusion model.")
    log = {"critic_loss": 0., "actor_loss":0., "target_q_mean": 0., "KL_divergence": 0.}
    
    if args.mode == 'pretrain': 
        for batch in loop_dataloader(dataloader):

            obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
            act = batch["act"].to(args.device)
            rew = batch["rew"].to(args.device)
            tml = batch["tml"].to(args.device)

            actor_loss = actor.loss(act, obs)
            actor.optimizer.zero_grad()
            actor_loss.backward()
            actor.optimizer.step()
            actor_lr_scheduler.step()
            pretrain_gradient_step += 1
            log["actor_loss"] += actor_loss.item()

            if pretrain_gradient_step % args.log_interval == 0:
                logger.info("Actor training gradient step:{}, Actor loss:{}".format(pretrain_gradient_step, log["actor_loss"]/args.log_interval))
                log["actor_loss"] = 0

            if pretrain_gradient_step >= args.gradient_steps:
                actor.save(save_path + f'diffusion_ckpt_{pretrain_gradient_step}.pt')
                break

    # ---------------------- Using SVGD to optimize the actions ----------------------
    svgd_gradient_step = 0
    actor.load(save_path + f'diffusion_ckpt_{args.gradient_steps}.pt')
    actor.eval()
    log = {"critic_loss": 0., "target_q_mean": 0., "KL_divergence": 0.}
    logger.info("Using SVGD to optimize the actions.")
    for batch in loop_dataloader(dataloader): 

        obs, next_obs = batch["obs"]["state"].to(args.device), batch["next_obs"]["state"].to(args.device)
        act = batch["act"].to(args.device)
        rew = batch["rew"].to(args.device)
        tml = batch["tml"].to(args.device)

        # generate behavioral actions
        prior = torch.zeros((args.batch_size, act_dim), device=args.device)
        a_0, _ = actor.sample(
                prior, solver=args.solver,
                n_samples=args.batch_size, sample_steps=args.sampling_steps, use_ema=False,
                temperature=1.0, condition_cfg=next_obs, w_cfg=1.0, requires_grad=True)

        # Critic Training
        current_q1, current_q2 = critic(obs, act)
        kernel = RBF(num_particles=args.training_num_particles, sigma=None, adaptive_sig=4, device=args.device)

        next_act, KL = svgd_update(a_0, next_obs, args.itr_num, args.svgd_step, args.training_num_particles, act_dim, critic, kernel, args.device)

        target_q = torch.min(*critic_target(next_obs, next_act)) + args.alpha * KL 
        # print("q:{},KL:{}".format(torch.min(*critic_target(next_obs, next_act)).mean(), KL.mean()))
        target_q = (rew + (1 - tml) * args.discount * target_q).detach()

        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        critic_optim.zero_grad()
        critic_loss.backward()
        critic_optim.step()
        critic_lr_scheduler.step()

        # -- ema
        if svgd_gradient_step % args.ema_update_interval == 0:
            if svgd_gradient_step >= 1000:
                actor.ema_update()
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(0.995 * param.data + (1 - 0.995) * target_param.data)

        # ----------- Logging ------------
        log["critic_loss"] += critic_loss.item()
        log["target_q_mean"] += target_q.mean().item()
        log["KL_divergence"] += KL.mean().item()

        if (svgd_gradient_step + 1) % args.log_interval == 0:
            log["gradient_steps"] = svgd_gradient_step + 1
            log["critic_loss"] /= args.log_interval
            log["target_q_mean"] /= args.log_interval
            log["KL_divergence"] /= args.log_interval
            logger.info("Training gradient step:{}, Critic loss:{}, Target q mean:{}, KL divergence:{}".format(log["gradient_steps"],log["critic_loss"],log["target_q_mean"],log["KL_divergence"]))
            log = {"critic_loss": 0., "target_q_mean": 0., "KL_divergence": 0.}

        # ----------- Inference ------------
        if (svgd_gradient_step + 1) % args.inference_interval == 0:
            actor.eval()
            critic.eval()
            critic_target.eval()

            env_eval = gym.vector.make(args.task.env_name, args.num_envs)
            normalizer = dataset.get_normalizer()
            episode_rewards = []

            prior = torch.zeros((args.num_envs * args.num_candidates, act_dim), device=args.device)
            for i in range(args.num_episodes):

                obs, ep_reward, cum_done, t = env_eval.reset(), 0., 0., 0

                while not np.all(cum_done) and t < 1000 + 1:
                    # normalize obs
                    obs = torch.tensor(normalizer.normalize(obs), device=args.device, dtype=torch.float32)
                    obs = obs.unsqueeze(1).repeat(1, args.num_candidates, 1).view(-1, obs_dim)
                    a_0, _ = actor.sample(
                    prior, solver=args.solver,
                    n_samples=args.num_envs * args.num_candidates, sample_steps=args.sampling_steps, use_ema=False,
                    temperature=1.0, condition_cfg=obs, w_cfg=1.0, requires_grad=True)

                    kernel = RBF(num_particles=args.inference_num_particles, sigma=None, adaptive_sig=4, device=args.device)
                    a_L, KL = svgd_update(a_0, obs, args.itr_num, args.svgd_step, args.inference_num_particles, act_dim, critic, kernel, args.device)

                    # resample
                    with torch.no_grad():
                        q = critic_target.q_min(obs, a_L)
                        q = q.view(-1, args.num_candidates, 1)
                        w = torch.softmax(q * args.task.weight_temperature, 1)
                        act = a_L.view(-1, args.num_candidates, act_dim)

                        indices = torch.multinomial(w.squeeze(-1), 1).squeeze(-1)
                        sampled_act = act[torch.arange(act.shape[0]), indices].cpu().numpy()

                    # step
                    obs, rew, done, info = env_eval.step(sampled_act)

                    t += 1
                    cum_done = done if cum_done is None else np.logical_or(cum_done, done)
                    ep_reward += (rew * (1 - cum_done)) if t < 1000 else rew

                    if np.all(cum_done):
                        break

                episode_rewards.append(ep_reward)

            episode_rewards = [list(map(lambda x: env.get_normalized_score(x), r)) for r in episode_rewards]
            episode_rewards = np.array(episode_rewards)
            logger.info("Inference gradient step:{}, mean:{}, std:{}".format(svgd_gradient_step + 1, np.mean(episode_rewards, -1), np.std(episode_rewards, -1)))

            critic.train()

        svgd_gradient_step += 1
        if svgd_gradient_step >= args.gradient_steps:
            torch.save({
                    "critic": critic.state_dict(),
                    "critic_target": critic_target.state_dict(),
                }, save_path + f"critic_ckpt_{svgd_gradient_step}.pt")
            break

if __name__ == "__main__":
    pipeline()
