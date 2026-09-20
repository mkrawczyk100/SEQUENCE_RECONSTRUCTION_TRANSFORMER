"""Etap drugi: dostrajanie algorytmem Proximal Policy Optimization.

Sformułowanie zadania jako procesu decyzyjnego:

    stan       - dotychczasowy prefiks sekwencji, czyli fragment
                 wejściowy wraz z tokenami wygenerowanymi do tej pory,
    akcja      - wybór jednego tokenu ze słownika,
    epizod     - kończy się na tokenie STOP albo po wyczerpaniu okna,
    nagroda    - wartość metryki podobieństwa między sceną odzyskaną
                 a referencyjną, przyznawana na końcu epizodu.

Nagroda jest rzadka, to znaczy przyznawana wyłącznie w kroku
terminalnym, ponieważ metryki podobieństwa są zdefiniowane dla
całych sekwencji, a nie dla pojedynczych tokenów. Wartościowanie
kroków pośrednich realizuje krytyk, czyli głowica wartości.

Do nagrody dodawana jest kara za rozbieżność względem polityki
odniesienia, będącej zamrożoną kopią modelu po etapie pierwszym.
Bez tej kary polityka szybko degeneruje się do sekwencji, które
maksymalizują metrykę, tracąc własności językowe wyuczone
w uczeniu nadzorowanym. Kara ma postać przybliżenia dywergencji
Kullbacka-Leiblera liczonego na poziomie pojedynczego tokenu.
"""

from __future__ import annotations

import copy
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from .config import ExperimentConfig
from .data import PAD, STOP, SceneDataset, make_prompt
from .evaluation import evaluate_model, extract_restored_scene, format_summary
from .metrics import REWARD_METRICS
from .model import SceneTransformer
from .utils import (
    HistoryLogger,
    build_model_from_checkpoint,
    disable_dropout,
    describe_device,
    resolve_device,
    save_checkpoint,
    set_seed,
)


# ---------------------------------------------------------------
# Estymacja przewagi (GAE), zaimplementowana bez zależności od torcha
# ---------------------------------------------------------------

def compute_gae(
    rewards: np.ndarray,
    values: np.ndarray,
    gamma: float,
    lam: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Uogólniona estymacja przewagi dla jednego epizodu.

    Argumenty odpowiadają kolejnym krokom epizodu: `rewards[t]` jest
    nagrodą za akcję w kroku t, a `values[t]` oszacowaniem wartości
    stanu poprzedzającego tę akcję. Wartość stanu następującego po
    ostatniej akcji wynosi zero, ponieważ epizod się kończy.

    Zwraca przewagi oraz zwroty, przy czym zwrot jest sumą przewagi
    i wartości stanu, zgodnie z definicją używaną jako cel regresji
    dla krytyka.
    """
    steps = len(rewards)
    advantages = np.zeros(steps, dtype=np.float64)
    running = 0.0
    for t in range(steps - 1, -1, -1):
        next_value = values[t + 1] if t + 1 < steps else 0.0
        delta = rewards[t] + gamma * next_value - values[t]
        running = delta + gamma * lam * running
        advantages[t] = running
    return advantages, advantages + values


def masked_mean(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Średnia po pozycjach wskazanych maską."""
    total = mask.sum()
    if total == 0:
        return tensor.sum() * 0.0
    return (tensor * mask).sum() / total


# ---------------------------------------------------------------
# Zbieranie doświadczenia
# ---------------------------------------------------------------

class Rollout:
    """Bufor doświadczenia zebranego w jednej iteracji PPO."""

    def __init__(self, **fields):
        self.__dict__.update(fields)


@torch.no_grad()
def collect_rollout(
    policy: SceneTransformer,
    reference: SceneTransformer,
    dataset: SceneDataset,
    config: ExperimentConfig,
    rng: random.Random,
    device: torch.device,
) -> Rollout:
    """Generuje epizody i wylicza wszystko, co potrzebne do aktualizacji."""
    ppo = config.ppo
    reward_fn = REWARD_METRICS[ppo.reward_metric]
    block_size = policy.config.block_size

    episodes: list[dict] = []

    # Fragmenty o jednakowej długości generowane są wsadowo.
    plan: dict[int, list[int]] = {}
    for _ in range(ppo.episodes_per_iteration):
        length = rng.randint(ppo.prompt_min, ppo.prompt_max)
        plan.setdefault(length, []).append(rng.randrange(dataset.num_scenes))

    for prompt_length, scene_ids in plan.items():
        references = [dataset.scenes[i] for i in scene_ids]
        prompts = [
            make_prompt(scene, prompt_length, rng) for scene in references
        ]
        prompt_tensor = torch.tensor(
            prompts, dtype=torch.long, device=device
        )
        limit = block_size - prompt_length - 1
        generated, _ = policy.generate(
            prompt_tensor,
            max_new_tokens=limit,
            temperature=ppo.temperature,
            top_k=ppo.top_k,
        )

        for row, scene in enumerate(references):
            tokens = generated[row].tolist()
            restored = extract_restored_scene(tokens, prompt_length)

            if restored is None:
                # Epizod bez tokenu STOP: kończy się na ostatnim
                # wygenerowanym tokenie mieszczącym się w oknie.
                reward = ppo.no_stop_penalty
                stop_found = False
                sequence = [t for t in tokens if t != PAD]
            else:
                reward = float(reward_fn(scene, restored))
                stop_found = True
                stop_index = tokens[prompt_length:].index(STOP)
                sequence = tokens[: prompt_length + stop_index + 1]

            if len(sequence) <= prompt_length:
                continue

            episodes.append({
                "tokens": sequence,
                "prompt_length": prompt_length,
                "reward": reward,
                "stop_found": stop_found,
                "restored_length": 0 if restored is None else len(restored),
            })

    if not episodes:
        raise RuntimeError("Nie zebrano żadnego poprawnego epizodu.")

    max_length = max(len(e["tokens"]) for e in episodes)
    num_episodes = len(episodes)

    tokens = torch.full(
        (num_episodes, max_length), PAD, dtype=torch.long, device=device
    )
    action_mask = torch.zeros(
        (num_episodes, max_length - 1), dtype=torch.bool, device=device
    )
    for row, episode in enumerate(episodes):
        length = len(episode["tokens"])
        tokens[row, :length] = torch.tensor(
            episode["tokens"], dtype=torch.long, device=device
        )
        first = episode["prompt_length"] - 1
        last = length - 2
        action_mask[row, first:last + 1] = True

    was_training = policy.training
    policy.eval()
    old_logprobs, values = _score_sequence(policy, tokens, with_value=True)
    ref_logprobs, _ = _score_sequence(reference, tokens, with_value=False)
    if was_training:
        policy.train()

    # Nagroda gęsta: kara za odejście od polityki odniesienia.
    kl_per_token = old_logprobs - ref_logprobs
    rewards = -ppo.kl_coef * kl_per_token
    rewards = rewards * action_mask

    for row, episode in enumerate(episodes):
        last = len(episode["tokens"]) - 2
        rewards[row, last] = rewards[row, last] + episode["reward"]

    advantages = torch.zeros_like(rewards)
    returns = torch.zeros_like(rewards)
    rewards_np = rewards.detach().cpu().numpy()
    values_np = values.detach().cpu().numpy()

    for row, episode in enumerate(episodes):
        first = episode["prompt_length"] - 1
        last = len(episode["tokens"]) - 2
        span = slice(first, last + 1)
        adv, ret = compute_gae(
            rewards_np[row, span].astype(np.float64),
            values_np[row, span].astype(np.float64),
            ppo.gamma,
            ppo.gae_lambda,
        )
        advantages[row, span] = torch.tensor(
            adv, dtype=advantages.dtype, device=device
        )
        returns[row, span] = torch.tensor(
            ret, dtype=returns.dtype, device=device
        )

    if ppo.whiten_advantages and action_mask.sum() > 1:
        selected = advantages[action_mask]
        advantages = torch.where(
            action_mask,
            (advantages - selected.mean()) / (selected.std() + 1e-8),
            advantages,
        )

    metric_rewards = np.array([e["reward"] for e in episodes])
    stats = {
        "episodes": num_episodes,
        "reward_mean": float(metric_rewards.mean()),
        "reward_std": float(metric_rewards.std()),
        "reward_max": float(metric_rewards.max()),
        "stop_rate": float(np.mean([e["stop_found"] for e in episodes])),
        "mean_episode_length": float(
            np.mean([len(e["tokens"]) - e["prompt_length"] for e in episodes])
        ),
        "mean_restored_length": float(
            np.mean([e["restored_length"] for e in episodes])
        ),
        "kl_to_reference": float(
            masked_mean(kl_per_token, action_mask).item()
        ),
    }

    return Rollout(
        tokens=tokens,
        action_mask=action_mask,
        old_logprobs=old_logprobs.detach(),
        values=values.detach(),
        advantages=advantages.detach(),
        returns=returns.detach(),
        stats=stats,
    )


def _score_sequence(
    model: SceneTransformer,
    tokens: torch.Tensor,
    with_value: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Log-prawdopodobieństwa tokenów oraz wartości stanów.

    Element o indeksie k odpowiada akcji wykonanej na pozycji k+1,
    czyli log-prawdopodobieństwu wygenerowania tokenu `tokens[k+1]`
    na podstawie prefiksu `tokens[0..k]`. Analogicznie `values[k]`
    jest oszacowaniem wartości stanu poprzedzającego tę akcję.
    """
    if with_value:
        logits, values = model.forward_with_value(tokens)
        values = values[:, :-1]
    else:
        logits, _ = model(tokens)
        values = torch.zeros(
            tokens.shape[0], tokens.shape[1] - 1, device=tokens.device
        )

    log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
    gathered = log_probs.gather(-1, tokens[:, 1:].unsqueeze(-1)).squeeze(-1)
    return gathered, values


# ---------------------------------------------------------------
# Aktualizacja polityki
# ---------------------------------------------------------------

def ppo_update(
    policy: SceneTransformer,
    optimizer: torch.optim.Optimizer,
    rollout: Rollout,
    config: ExperimentConfig,
    rng: random.Random,
) -> dict[str, float]:
    """Wykonuje kilka epok optymalizacji na zebranym buforze."""
    ppo = config.ppo
    num_episodes = rollout.tokens.shape[0]
    order = list(range(num_episodes))

    logs = {
        "policy_loss": [], "value_loss": [], "entropy": [],
        "approx_kl": [], "clip_fraction": [], "grad_norm": [],
    }

    for _ in range(ppo.ppo_epochs):
        rng.shuffle(order)
        for start in range(0, num_episodes, ppo.minibatch_size):
            index = order[start:start + ppo.minibatch_size]
            if not index:
                continue
            rows = torch.tensor(index, device=rollout.tokens.device)

            tokens = rollout.tokens[rows]
            mask = rollout.action_mask[rows]
            if mask.sum() == 0:
                continue

            old_logprobs = rollout.old_logprobs[rows]
            old_values = rollout.values[rows]
            advantages = rollout.advantages[rows]
            returns = rollout.returns[rows]

            logits, values = policy.forward_with_value(tokens)
            values = values[:, :-1]
            log_probs = F.log_softmax(logits[:, :-1, :].float(), dim=-1)
            new_logprobs = log_probs.gather(
                -1, tokens[:, 1:].unsqueeze(-1)
            ).squeeze(-1)

            ratio = torch.exp(new_logprobs - old_logprobs)
            surrogate_1 = ratio * advantages
            surrogate_2 = torch.clamp(
                ratio, 1.0 - ppo.clip_range, 1.0 + ppo.clip_range
            ) * advantages
            policy_loss = -masked_mean(
                torch.min(surrogate_1, surrogate_2), mask
            )

            clipped_values = old_values + torch.clamp(
                values - old_values,
                -ppo.value_clip_range,
                ppo.value_clip_range,
            )
            value_loss = 0.5 * masked_mean(
                torch.max(
                    (values - returns) ** 2,
                    (clipped_values - returns) ** 2,
                ),
                mask,
            )

            entropy = -(log_probs.exp() * log_probs).sum(-1)
            entropy_mean = masked_mean(entropy, mask)

            loss = (
                policy_loss
                + ppo.value_coef * value_loss
                - ppo.entropy_coef * entropy_mean
            )

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), ppo.grad_clip
            )
            optimizer.step()

            with torch.no_grad():
                approx_kl = masked_mean(
                    old_logprobs - new_logprobs, mask
                )
                clip_fraction = masked_mean(
                    ((ratio - 1.0).abs() > ppo.clip_range).float(), mask
                )

            logs["policy_loss"].append(policy_loss.item())
            logs["value_loss"].append(value_loss.item())
            logs["entropy"].append(entropy_mean.item())
            logs["approx_kl"].append(approx_kl.item())
            logs["clip_fraction"].append(clip_fraction.item())
            logs["grad_norm"].append(float(grad_norm))

    return {key: float(np.mean(v)) if v else 0.0 for key, v in logs.items()}


# ---------------------------------------------------------------
# Pętla główna
# ---------------------------------------------------------------

def train_ppo(
    config: ExperimentConfig,
    sft_checkpoint: str,
) -> str:
    """Dostraja model po etapie pierwszym wybraną metryką jako nagrodą."""
    if config.ppo.reward_metric not in REWARD_METRICS:
        raise ValueError(
            f"Nieznana metryka nagrody: {config.ppo.reward_metric}. "
            f"Dostępne: {sorted(REWARD_METRICS)}"
        )

    set_seed(config.seed)
    device = resolve_device(config.device)
    os.makedirs(config.output_dir, exist_ok=True)

    policy, sft_config, _ = build_model_from_checkpoint(
        sft_checkpoint, device, with_value_head=True
    )
    # Konfiguracja architektury i danych musi pochodzić z etapu
    # pierwszego, żeby polityka początkowa pasowała do modelu.
    config.model = sft_config.model
    config.data = sft_config.data

    # Dropout musi zostać wyłączony na czas PPO, inaczej iloraz
    # prawdopodobieństw nie wynosi jeden przed pierwszą aktualizacją.
    disabled = disable_dropout(policy)
    print(f"Wyłączono dropout w {disabled} warstwach.")

    reference = copy.deepcopy(policy)
    reference.value_head = None
    for parameter in reference.parameters():
        parameter.requires_grad_(False)
    reference.eval()

    dataset = SceneDataset(config.data.path)
    rng = random.Random(config.seed)

    optimizer = torch.optim.AdamW(
        [p for p in policy.parameters() if p.requires_grad],
        lr=config.ppo.learning_rate,
        betas=(0.9, 0.95),
    )

    logger = HistoryLogger(config.output_dir, "history_ppo")
    config.to_json(os.path.join(config.output_dir, "config.json"))

    print(f"Urządzenie: {describe_device(device)}")
    print(f"Polityka początkowa: {sft_checkpoint}")
    print(f"Metryka nagrody: {config.ppo.reward_metric}")
    print(f"Kara KL: {config.ppo.kl_coef}\n")

    baseline = evaluate_model(
        policy, dataset, device,
        num_scenes=config.ppo.eval_samples,
        prompt_length=config.train.prompt_length,
        seed=config.eval_seed,
        greedy=config.ppo.greedy_eval,
    )
    print(f"Punkt odniesienia (po SFT): {format_summary(baseline)}\n")
    logger.log({"iteration": -1, **{f"eval_{k}": v
                                    for k, v in baseline.items()}})

    best_score = -float("inf")
    for iteration in range(config.ppo.iterations):
        policy.train()
        rollout = collect_rollout(
            policy, reference, dataset, config, rng, device
        )
        update_stats = ppo_update(policy, optimizer, rollout, config, rng)

        record = {
            "iteration": iteration,
            "learning_rate": config.ppo.learning_rate,
            **rollout.stats,
            **update_stats,
        }

        if iteration % config.ppo.eval_every == 0:
            summary = evaluate_model(
                policy, dataset, device,
                num_scenes=config.ppo.eval_samples,
                prompt_length=config.train.prompt_length,
                seed=config.eval_seed,
                greedy=config.ppo.greedy_eval,
            )
            record.update({f"eval_{k}": v for k, v in summary.items()})
            print(f"Iteracja {iteration:4d} | nagroda "
                  f"{rollout.stats['reward_mean']:.4f} | KL "
                  f"{rollout.stats['kl_to_reference']:.4f} | "
                  f"{format_summary(summary)}")

            score = summary["content_f1"] + summary["order_tau"]
            if score > best_score:
                best_score = score
                save_checkpoint(
                    os.path.join(config.output_dir, "best.pt"),
                    policy, optimizer, config, iteration, logger.records,
                    extra={"best_target_score": score},
                )
        else:
            print(f"Iteracja {iteration:4d} | nagroda "
                  f"{rollout.stats['reward_mean']:.4f} "
                  f"(sd {rollout.stats['reward_std']:.3f}) | KL "
                  f"{rollout.stats['kl_to_reference']:.4f} | STOP "
                  f"{rollout.stats['stop_rate']:.3f} | entropia "
                  f"{update_stats['entropy']:.3f}")

        logger.log(record)
        logger.flush_csv()
        save_checkpoint(
            os.path.join(config.output_dir, "last.pt"),
            policy, optimizer, config, iteration, logger.records,
        )

    return os.path.join(config.output_dir, "best.pt")
