"""Ghost Agent Replacement (GAR) proof-of-concept test.

Validates that a frozen MAPPO 1v1 checkpoint (trained with 2 agents) can be
deployed with only 1 real agent while the other slot is filled by a passive
(noop) decision-maker.

Setup:
    - Environment: MosaicMultiGrid-AmericanFootball-1v1-v0 (view_size=7)
    - agent_0 (Green): MAPPO policy (real -- actions go to env)
    - agent_1 (Blue):  Ghost RL inference (computed, discarded) +
                       Passive worker (noop action 0 used in env)

The MAPPO checkpoint was trained with parameter_sharing=True on 2 agents.
The shared network requires a one-hot agents_id input to produce any output.
Ghost inference keeps the policy's architectural contract satisfied while
the passive agent stands still.

Key assertions:
    1. The MAPPO agent produces valid peaked actions (not uniform/random)
    2. The passive agent never moves (always action 0 = noop)
    3. The MAPPO agent scores touchdowns against the stationary opponent
    4. Ghost inference does not crash or produce NaN
"""

import logging
import os
from pathlib import Path

import numpy as np
import pytest
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CHECKPOINT_PATH = Path(
    "var/trainer/runs/run2_mappo_af_1v1/checkpoints/"
    "american_football_1v1/seed_1_2026_0408_221224/final_train_model.pth"
)
ENV_ID = "MosaicMultiGrid-AmericanFootball-1v1-v0"
VIEW_SIZE = 7
OBS_DIM = VIEW_SIZE * VIEW_SIZE * 3  # 147
N_AGENTS = 2
N_ACTIONS = 8
NOOP_ACTION = 0
N_EPISODES = 5
N_EPISODES_LONG = 20
MAX_STEPS = 300
SEED = 42


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _project_root() -> Path:
    """Walk up from this file to find the project root (contains .git)."""
    p = Path(__file__).resolve()
    for parent in [p] + list(p.parents):
        if (parent / ".git").exists():
            return parent
    return Path.cwd()


@pytest.fixture(scope="module")
def project_root():
    return _project_root()


@pytest.fixture(scope="module")
def checkpoint(project_root):
    path = project_root / CHECKPOINT_PATH
    if not path.exists():
        pytest.skip(f"MAPPO AF 1v1 checkpoint not found: {path}")
    return path


@pytest.fixture(scope="module")
def mappo_agent(checkpoint):
    """Load the MAPPO agent via XuanCe's get_runner."""
    from types import SimpleNamespace

    # The checkpoint was trained with MOSAIC_VIEW_SIZE=7 (obs_dim=147).
    # XuanCe's env wrapper reads this to set view_size on gym.make().
    os.environ["MOSAIC_VIEW_SIZE"] = str(VIEW_SIZE)

    from xuance import get_runner
    from xuance_worker.xuance_shims import apply_shims

    apply_shims()

    parser_args = SimpleNamespace()
    parser_args.dl_toolbox = "torch"
    parser_args.device = "cpu"
    parser_args.parallels = 1
    parser_args.running_steps = 1

    from xuance_worker.runtime import _resolve_custom_config_path

    config_path = _resolve_custom_config_path(
        method="mappo",
        env="multigrid",
        env_id="american_football_1v1",
        num_groups=None,
        config_path=None,
    )

    runner = get_runner(
        algo="mappo",
        env="multigrid",
        env_id="american_football_1v1",
        config_path=config_path,
        parser_args=parser_args,
    )

    agent = runner.agent
    state_dict = torch.load(str(checkpoint), map_location="cpu")
    agent.policy.load_state_dict(state_dict)
    agent.policy.eval()

    assert hasattr(agent, "n_agents"), "Agent should expose n_agents"
    assert agent.n_agents == N_AGENTS, f"Expected {N_AGENTS} agents, got {agent.n_agents}"
    assert getattr(agent, "use_parameter_sharing", False), "Expected parameter_sharing=True"

    return agent


@pytest.fixture(scope="module")
def env():
    """Create the AF 1v1 environment with view_size=7."""
    import mosaic_multigrid.envs  # noqa: F401
    import gymnasium as gym

    e = gym.make(ENV_ID, view_size=VIEW_SIZE, render_mode=None)
    yield e
    e.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_mappo_action(agent, obs_flat: np.ndarray, player_id: str, deterministic: bool = True) -> int:
    """Query the MAPPO policy for a single agent's action.

    This mirrors xuance_worker/runtime.py:_get_marl_action() but stripped
    down to the essentials for testing.
    """
    agent_key = agent.model_keys[0]
    agent_idx = int(player_id.split("_")[-1])

    obs_tensor = torch.FloatTensor(obs_flat).unsqueeze(0)  # (1, obs_dim)
    agents_id = torch.zeros(1, N_AGENTS, dtype=torch.float32)
    agents_id[0, agent_idx] = 1.0

    with torch.no_grad():
        _, pi_dists = agent.policy(
            observation={agent_key: obs_tensor},
            agent_ids=agents_id,
            agent_key=agent_key,
        )

    dist = pi_dists[agent_key]
    if deterministic:
        action = dist.deterministic_sample()
    else:
        action = dist.stochastic_sample()

    probs = dist.distribution.probs[0].cpu().numpy()
    return int(action.cpu().item()), probs


def flatten_obs(obs_dict: dict) -> np.ndarray:
    """Flatten a single agent's observation dict to a 1D array.

    The MAPPO policy expects the flattened 'image' field (7,7,3) -> (147,).
    """
    image = obs_dict["image"]
    return image.flatten().astype(np.float32)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestGhostAgentReplacement:
    """Proof-of-concept: MAPPO 1/2 real + 1/2 ghost against passive opponent."""

    def test_mappo_produces_peaked_actions(self, mappo_agent, env):
        """The trained policy should produce peaked (non-uniform) action distributions."""
        obs, _ = env.reset(seed=SEED)
        obs_flat = flatten_obs(obs[0])

        action, probs = get_mappo_action(mappo_agent, obs_flat, "agent_0")

        # A trained policy should have max_prob >> 1/N_ACTIONS (0.125)
        max_prob = float(probs.max())
        assert max_prob > 0.2, (
            f"Policy looks untrained: max_prob={max_prob:.3f} "
            f"(uniform would be {1/N_ACTIONS:.3f})"
        )
        assert 0 <= action < N_ACTIONS, f"Invalid action: {action}"
        logger.info("Action=%d, max_prob=%.3f, probs=%s", action, max_prob, probs)

    def test_ghost_inference_produces_valid_output(self, mappo_agent, env):
        """Ghost inference for agent_1 should not crash or produce NaN."""
        obs, _ = env.reset(seed=SEED)
        obs_flat = flatten_obs(obs[1])

        ghost_action, ghost_probs = get_mappo_action(mappo_agent, obs_flat, "agent_1")

        assert 0 <= ghost_action < N_ACTIONS, f"Invalid ghost action: {ghost_action}"
        assert not np.any(np.isnan(ghost_probs)), "Ghost probs contain NaN"
        assert abs(ghost_probs.sum() - 1.0) < 1e-5, "Ghost probs don't sum to 1"
        logger.info("Ghost action=%d, probs=%s", ghost_action, ghost_probs)

    def test_ghost_does_not_affect_real_agent(self, mappo_agent, env):
        """Querying the ghost (agent_1) should not change agent_0's output.

        For feed-forward MAPPO (use_rnn=False), each forward pass is
        independent -- no hidden state carries over between calls.
        """
        obs, _ = env.reset(seed=SEED)
        obs0_flat = flatten_obs(obs[0])

        # Query agent_0 WITHOUT ghost first
        action_before, probs_before = get_mappo_action(mappo_agent, obs0_flat, "agent_0")

        # Now query ghost (agent_1)
        obs1_flat = flatten_obs(obs[1])
        _, _ = get_mappo_action(mappo_agent, obs1_flat, "agent_1")

        # Query agent_0 again AFTER ghost
        action_after, probs_after = get_mappo_action(mappo_agent, obs0_flat, "agent_0")

        # Must be identical (feed-forward: no state)
        np.testing.assert_array_almost_equal(
            probs_before, probs_after,
            decimal=6,
            err_msg="Ghost inference changed real agent's output distribution",
        )
        assert action_before == action_after, (
            f"Ghost inference changed real agent's action: {action_before} -> {action_after}"
        )

    def test_gar_episode_mappo_vs_passive(self, mappo_agent, env):
        """Run full episodes: agent_0=MAPPO (real), agent_1=passive (noop).

        Ghost inference is computed for agent_1 to satisfy the policy contract
        but discarded. The passive agent always submits action 0 (noop).

        The MAPPO agent should score touchdowns against a stationary opponent.
        """
        total_green_reward = 0.0
        total_blue_reward = 0.0
        green_wins = 0
        episodes_completed = 0

        for ep in range(N_EPISODES):
            obs, _ = env.reset(seed=SEED + ep)
            ep_green_reward = 0.0
            ep_blue_reward = 0.0

            for step in range(MAX_STEPS):
                # --- Agent 0 (Green): MAPPO real action ---
                obs0_flat = flatten_obs(obs[0])
                real_action, _ = get_mappo_action(
                    mappo_agent, obs0_flat, "agent_0", deterministic=True,
                )

                # --- Agent 1 (Blue): Ghost inference + passive override ---
                obs1_flat = flatten_obs(obs[1])
                ghost_action, _ = get_mappo_action(
                    mappo_agent, obs1_flat, "agent_1", deterministic=True,
                )
                # Ghost action is DISCARDED. Passive noop is used instead.
                passive_action = NOOP_ACTION

                # --- Step environment ---
                actions = {0: real_action, 1: passive_action}
                obs, rewards, terminateds, truncateds, infos = env.step(actions)

                ep_green_reward += rewards[0]
                ep_blue_reward += rewards[1]

                if any(terminateds.values()) or any(truncateds.values()):
                    break

            total_green_reward += ep_green_reward
            total_blue_reward += ep_blue_reward
            if ep_green_reward > ep_blue_reward:
                green_wins += 1
            episodes_completed += 1

            logger.info(
                "Episode %d: green=%.1f blue=%.1f steps=%d ghost_last=%d",
                ep, ep_green_reward, ep_blue_reward, step + 1, ghost_action,
            )

        # --- Assertions ---
        avg_green = total_green_reward / episodes_completed
        avg_blue = total_blue_reward / episodes_completed

        logger.info(
            "GAR results: %d episodes, green_avg=%.2f, blue_avg=%.2f, "
            "green_wins=%d/%d",
            episodes_completed, avg_green, avg_blue,
            green_wins, episodes_completed,
        )

        # MAPPO agent should dominate a stationary opponent
        assert avg_green > avg_blue, (
            f"MAPPO agent should beat passive opponent: "
            f"green_avg={avg_green:.2f} vs blue_avg={avg_blue:.2f}"
        )
        assert green_wins > 0, (
            f"MAPPO agent should win at least 1 episode against passive "
            f"(won {green_wins}/{episodes_completed})"
        )

    def test_gar_vs_selfplay_action_divergence(self, mappo_agent, env):
        """Compare GAR (MAPPO vs passive) to self-play (MAPPO vs MAPPO).

        In self-play, agent_1 uses its MAPPO actions.
        In GAR, agent_1 uses noop.
        The resulting trajectories should diverge, proving the ghost's
        action is truly discarded and the passive action is used.
        """
        # --- Self-play trajectory ---
        obs_sp, _ = env.reset(seed=SEED)
        sp_states = []
        for step in range(20):
            obs0_flat = flatten_obs(obs_sp[0])
            obs1_flat = flatten_obs(obs_sp[1])
            a0, _ = get_mappo_action(mappo_agent, obs0_flat, "agent_0", deterministic=True)
            a1, _ = get_mappo_action(mappo_agent, obs1_flat, "agent_1", deterministic=True)
            obs_sp, _, terms, truncs, _ = env.step({0: a0, 1: a1})
            sp_states.append(obs_sp[0]["image"].flatten().copy())
            if any(terms.values()) or any(truncs.values()):
                break

        # --- GAR trajectory (same seed) ---
        obs_gar, _ = env.reset(seed=SEED)
        gar_states = []
        for step in range(20):
            obs0_flat = flatten_obs(obs_gar[0])
            obs1_flat = flatten_obs(obs_gar[1])
            a0, _ = get_mappo_action(mappo_agent, obs0_flat, "agent_0", deterministic=True)
            # Ghost: compute but discard
            _, _ = get_mappo_action(mappo_agent, obs1_flat, "agent_1", deterministic=True)
            obs_gar, _, terms, truncs, _ = env.step({0: a0, 1: NOOP_ACTION})
            gar_states.append(obs_gar[0]["image"].flatten().copy())
            if any(terms.values()) or any(truncs.values()):
                break

        # Trajectories should diverge (agent_1 moving vs standing still
        # changes the game state observed by agent_0)
        min_len = min(len(sp_states), len(gar_states))
        assert min_len >= 2, "Need at least 2 steps for comparison"

        diverged = False
        for i in range(min_len):
            if not np.array_equal(sp_states[i], gar_states[i]):
                diverged = True
                logger.info("Trajectories diverged at step %d", i)
                break

        assert diverged, (
            "Self-play and GAR trajectories are identical -- "
            "ghost action discard may not be working "
            "(or agent_1's MAPPO action happens to be noop every step)"
        )

    def test_gar_mappo_vs_passive_long(self, mappo_agent, env):
        """Extended GAR run: MAPPO vs passive over 20 episodes.

        Collects per-episode stats to show consistency of the trained
        policy against a stationary opponent across many seeds.
        """
        results = []

        for ep in range(N_EPISODES_LONG):
            obs, _ = env.reset(seed=SEED + ep * 7)  # spread seeds
            ep_green = 0.0
            ep_blue = 0.0
            ep_steps = 0

            for step in range(MAX_STEPS):
                obs0_flat = flatten_obs(obs[0])
                real_action, _ = get_mappo_action(
                    mappo_agent, obs0_flat, "agent_0", deterministic=True,
                )

                obs1_flat = flatten_obs(obs[1])
                # Ghost: feed the policy, discard the output
                _, _ = get_mappo_action(
                    mappo_agent, obs1_flat, "agent_1", deterministic=True,
                )

                actions = {0: real_action, 1: NOOP_ACTION}
                obs, rewards, terminateds, truncateds, infos = env.step(actions)

                ep_green += rewards[0]
                ep_blue += rewards[1]
                ep_steps = step + 1

                if any(terminateds.values()) or any(truncateds.values()):
                    break

            results.append({
                "green": ep_green, "blue": ep_blue, "steps": ep_steps,
                "winner": "green" if ep_green > ep_blue else ("blue" if ep_blue > ep_green else "draw"),
            })

        green_rewards = [r["green"] for r in results]
        blue_rewards = [r["blue"] for r in results]
        green_wins = sum(1 for r in results if r["winner"] == "green")
        draws = sum(1 for r in results if r["winner"] == "draw")
        avg_steps = np.mean([r["steps"] for r in results])

        logger.info(
            "MAPPO vs Passive (%d episodes): green_avg=%.2f blue_avg=%.2f "
            "green_wins=%d draws=%d avg_steps=%.0f",
            N_EPISODES_LONG, np.mean(green_rewards), np.mean(blue_rewards),
            green_wins, draws, avg_steps,
        )
        for i, r in enumerate(results):
            logger.info(
                "  ep %2d: green=%5.1f blue=%5.1f steps=%3d  %s",
                i, r["green"], r["blue"], r["steps"], r["winner"],
            )

        # Over 20 episodes the MAPPO agent should convincingly beat passive
        assert np.mean(green_rewards) > 0, (
            f"MAPPO should have positive avg reward vs passive: {np.mean(green_rewards):.2f}"
        )
        assert green_wins >= N_EPISODES_LONG // 3, (
            f"MAPPO should win at least 1/3 of episodes vs passive: "
            f"won {green_wins}/{N_EPISODES_LONG}"
        )

    def test_gar_mappo_vs_random_long(self, mappo_agent, env):
        """Extended GAR run: MAPPO vs random over 20 episodes.

        agent_0 (Green) = MAPPO real actions
        agent_1 (Blue)  = Ghost RL (discarded) + random actions (used)

        This is the more interesting GAR test: the opponent actually moves
        and takes random actions, creating a dynamic (if chaotic) game.
        The MAPPO agent should still dominate a random opponent.
        """
        rng = np.random.RandomState(SEED)
        results = []

        for ep in range(N_EPISODES_LONG):
            obs, _ = env.reset(seed=SEED + ep * 7)
            ep_green = 0.0
            ep_blue = 0.0
            ep_steps = 0

            for step in range(MAX_STEPS):
                # --- Agent 0 (Green): MAPPO real action ---
                obs0_flat = flatten_obs(obs[0])
                real_action, _ = get_mappo_action(
                    mappo_agent, obs0_flat, "agent_0", deterministic=True,
                )

                # --- Agent 1 (Blue): Ghost + random replacement ---
                obs1_flat = flatten_obs(obs[1])
                ghost_action, _ = get_mappo_action(
                    mappo_agent, obs1_flat, "agent_1", deterministic=True,
                )
                # Ghost action DISCARDED. Random action used instead.
                random_action = rng.randint(0, N_ACTIONS)

                actions = {0: real_action, 1: random_action}
                obs, rewards, terminateds, truncateds, infos = env.step(actions)

                ep_green += rewards[0]
                ep_blue += rewards[1]
                ep_steps = step + 1

                if any(terminateds.values()) or any(truncateds.values()):
                    break

            results.append({
                "green": ep_green, "blue": ep_blue, "steps": ep_steps,
                "winner": "green" if ep_green > ep_blue else ("blue" if ep_blue > ep_green else "draw"),
            })

        green_rewards = [r["green"] for r in results]
        blue_rewards = [r["blue"] for r in results]
        green_wins = sum(1 for r in results if r["winner"] == "green")
        blue_wins = sum(1 for r in results if r["winner"] == "blue")
        draws = sum(1 for r in results if r["winner"] == "draw")
        avg_steps = np.mean([r["steps"] for r in results])

        logger.info(
            "MAPPO vs Random (%d episodes): green_avg=%.2f blue_avg=%.2f "
            "green_wins=%d blue_wins=%d draws=%d avg_steps=%.0f",
            N_EPISODES_LONG, np.mean(green_rewards), np.mean(blue_rewards),
            green_wins, blue_wins, draws, avg_steps,
        )
        for i, r in enumerate(results):
            logger.info(
                "  ep %2d: green=%5.1f blue=%5.1f steps=%3d  %s",
                i, r["green"], r["blue"], r["steps"], r["winner"],
            )

        # MAPPO should beat random, but random can occasionally stumble
        # into the end zone, so we allow some blue wins
        assert np.mean(green_rewards) > np.mean(blue_rewards), (
            f"MAPPO should outperform random on average: "
            f"green={np.mean(green_rewards):.2f} vs blue={np.mean(blue_rewards):.2f}"
        )
        assert green_wins > blue_wins, (
            f"MAPPO should win more than random: "
            f"green_wins={green_wins} vs blue_wins={blue_wins}"
        )

    def test_gar_mappo_vs_selfplay_long(self, mappo_agent, env):
        """Extended self-play baseline: MAPPO vs MAPPO over 20 episodes.

        Both agents use the same trained policy. This serves as the
        upper-bound comparison for the GAR tests:
        - vs passive: MAPPO should dominate (easy)
        - vs random:  MAPPO should dominate (medium)
        - vs self:    roughly even (hard -- same policy, symmetric game)
        """
        results = []

        for ep in range(N_EPISODES_LONG):
            obs, _ = env.reset(seed=SEED + ep * 7)
            ep_green = 0.0
            ep_blue = 0.0
            ep_steps = 0

            for step in range(MAX_STEPS):
                obs0_flat = flatten_obs(obs[0])
                obs1_flat = flatten_obs(obs[1])
                a0, _ = get_mappo_action(mappo_agent, obs0_flat, "agent_0", deterministic=True)
                a1, _ = get_mappo_action(mappo_agent, obs1_flat, "agent_1", deterministic=True)

                obs, rewards, terminateds, truncateds, infos = env.step({0: a0, 1: a1})

                ep_green += rewards[0]
                ep_blue += rewards[1]
                ep_steps = step + 1

                if any(terminateds.values()) or any(truncateds.values()):
                    break

            results.append({
                "green": ep_green, "blue": ep_blue, "steps": ep_steps,
                "winner": "green" if ep_green > ep_blue else ("blue" if ep_blue > ep_green else "draw"),
            })

        green_rewards = [r["green"] for r in results]
        blue_rewards = [r["blue"] for r in results]
        green_wins = sum(1 for r in results if r["winner"] == "green")
        blue_wins = sum(1 for r in results if r["winner"] == "blue")
        draws = sum(1 for r in results if r["winner"] == "draw")
        avg_steps = np.mean([r["steps"] for r in results])

        logger.info(
            "MAPPO vs MAPPO self-play (%d episodes): green_avg=%.2f blue_avg=%.2f "
            "green_wins=%d blue_wins=%d draws=%d avg_steps=%.0f",
            N_EPISODES_LONG, np.mean(green_rewards), np.mean(blue_rewards),
            green_wins, blue_wins, draws, avg_steps,
        )
        for i, r in enumerate(results):
            logger.info(
                "  ep %2d: green=%5.1f blue=%5.1f steps=%3d  %s",
                i, r["green"], r["blue"], r["steps"], r["winner"],
            )

        # Self-play with parameter_sharing=True and a symmetric game
        # should be roughly balanced (same policy, seed-dependent asymmetry)
        total_games = green_wins + blue_wins
        if total_games > 0:
            win_ratio = green_wins / total_games
            # Allow range [0.15, 0.85] -- some asymmetry from initial positions
            assert 0.10 <= win_ratio <= 0.90, (
                f"Self-play should be roughly balanced: "
                f"green={green_wins} blue={blue_wins} ratio={win_ratio:.2f}"
            )


    def test_gar_mappo_full_2of2_both_real(self, mappo_agent, env):
        """Full MAPPO 2/2: both agents are REAL, no ghosts.

        This is the standard linked deployment where both agent_0 and
        agent_1 use the same MAPPO checkpoint and both actions go to the
        environment. No ghost inference, no action replacement.

        This validates that the same policy + same get_mappo_action()
        helper works identically to the self-play baseline, and serves
        as the reference point: if LinkGroup has all RL agents, every
        agent is real and the full trained team plays together.
        """
        results = []

        for ep in range(N_EPISODES_LONG):
            obs, _ = env.reset(seed=SEED + ep * 7)
            ep_green = 0.0
            ep_blue = 0.0
            ep_steps = 0

            for step in range(MAX_STEPS):
                # --- Both agents are REAL: full MAPPO 2/2 ---
                obs0_flat = flatten_obs(obs[0])
                obs1_flat = flatten_obs(obs[1])
                a0, probs0 = get_mappo_action(
                    mappo_agent, obs0_flat, "agent_0", deterministic=True,
                )
                a1, probs1 = get_mappo_action(
                    mappo_agent, obs1_flat, "agent_1", deterministic=True,
                )

                # Both actions go to the environment (no ghost, no discard)
                obs, rewards, terminateds, truncateds, infos = env.step({0: a0, 1: a1})

                ep_green += rewards[0]
                ep_blue += rewards[1]
                ep_steps = step + 1

                if any(terminateds.values()) or any(truncateds.values()):
                    break

            results.append({
                "green": ep_green, "blue": ep_blue, "steps": ep_steps,
                "winner": "green" if ep_green > ep_blue
                    else ("blue" if ep_blue > ep_green else "draw"),
            })

        green_rewards = [r["green"] for r in results]
        blue_rewards = [r["blue"] for r in results]
        green_wins = sum(1 for r in results if r["winner"] == "green")
        blue_wins = sum(1 for r in results if r["winner"] == "blue")
        draws = sum(1 for r in results if r["winner"] == "draw")
        avg_steps = np.mean([r["steps"] for r in results])

        logger.info(
            "MAPPO 2/2 full real (%d episodes): green_avg=%.2f blue_avg=%.2f "
            "green_wins=%d blue_wins=%d draws=%d avg_steps=%.0f",
            N_EPISODES_LONG, np.mean(green_rewards), np.mean(blue_rewards),
            green_wins, blue_wins, draws, avg_steps,
        )
        for i, r in enumerate(results):
            logger.info(
                "  ep %2d: green=%5.1f blue=%5.1f steps=%3d  %s",
                i, r["green"], r["blue"], r["steps"], r["winner"],
            )

        # --- Assertions ---
        # Both agents are equally skilled (same shared policy).
        # In a symmetric zero-sum game, neither side should dominate.
        total_decisive = green_wins + blue_wins
        if total_decisive > 0:
            win_ratio = green_wins / total_decisive
            assert 0.10 <= win_ratio <= 0.90, (
                f"Full MAPPO 2/2 should be roughly balanced: "
                f"green={green_wins} blue={blue_wins} ratio={win_ratio:.2f}"
            )

        # Both agents should score something over 20 episodes
        # (not all draws, that would mean the policy is broken)
        assert np.mean(green_rewards) > 0 or np.mean(blue_rewards) > 0, (
            "At least one side should score across 20 episodes"
        )

        # Rewards should be comparable (neither side shut out)
        if total_decisive >= 4:
            ratio = min(np.sum(green_rewards), np.sum(blue_rewards)) / \
                    max(np.sum(green_rewards), np.sum(blue_rewards))
            assert ratio > 0.15, (
                f"Reward ratio too lopsided for symmetric self-play: {ratio:.2f}"
            )


class TestGhostDetectionLogic:
    """Unit tests for the ghost detection rule:
    'agent in LinkGroup but worker_type != rl' => ghost.
    """

    def test_ghost_detection_basic(self):
        """LLM agent in a link group should be detected as ghost."""
        from gym_gui.services.operator import (
            LinkGroup,
            OperatorConfig,
            WorkerAssignment,
        )

        config = OperatorConfig.multi_agent(
            operator_id="test_gar",
            display_name="GAR Test",
            env_name="mosaic_multigrid",
            task=ENV_ID,
            player_workers={
                "agent_0": WorkerAssignment("xuance_worker", "rl", {}),
                "agent_1": WorkerAssignment("passive_worker", "passive", {}),
            },
            link_groups={
                "test_link_0": LinkGroup(
                    group_id="test_link_0",
                    primary_agent="agent_0",
                    linked_agents=["agent_1"],
                    policy_path=str(CHECKPOINT_PATH),
                    algorithm="mappo",
                ),
            },
        )

        # Derive ghost agents: in link group but worker_type != "rl"
        ghosts = {}
        for group_id, group in config.link_groups.items():
            for agent_id in group.all_agents():
                assignment = config.workers.get(agent_id)
                if assignment and assignment.worker_type != "rl":
                    ghosts[agent_id] = group_id

        assert ghosts == {"agent_1": "test_link_0"}
        assert "agent_0" not in ghosts

    def test_no_ghosts_when_all_rl(self):
        """All RL agents in a link group => no ghosts."""
        from gym_gui.services.operator import (
            LinkGroup,
            OperatorConfig,
            WorkerAssignment,
        )

        config = OperatorConfig.multi_agent(
            operator_id="test_no_ghost",
            display_name="No Ghost Test",
            env_name="mosaic_multigrid",
            task=ENV_ID,
            player_workers={
                "agent_0": WorkerAssignment("xuance_worker", "rl", {}),
                "agent_1": WorkerAssignment("xuance_worker", "rl", {}),
            },
            link_groups={
                "link_0": LinkGroup(
                    group_id="link_0",
                    primary_agent="agent_0",
                    linked_agents=["agent_1"],
                    policy_path=str(CHECKPOINT_PATH),
                    algorithm="mappo",
                ),
            },
        )

        ghosts = {}
        for group_id, group in config.link_groups.items():
            for agent_id in group.all_agents():
                assignment = config.workers.get(agent_id)
                if assignment and assignment.worker_type != "rl":
                    ghosts[agent_id] = group_id

        assert ghosts == {}

    def test_no_ghosts_without_link_group(self):
        """Agents not in any link group are never ghosts, even if non-RL."""
        from gym_gui.services.operator import (
            OperatorConfig,
            WorkerAssignment,
        )

        config = OperatorConfig.multi_agent(
            operator_id="test_unlinked",
            display_name="Unlinked Test",
            env_name="mosaic_multigrid",
            task=ENV_ID,
            player_workers={
                "agent_0": WorkerAssignment("xuance_worker", "rl", {}),
                "agent_1": WorkerAssignment("passive_worker", "passive", {}),
            },
        )

        ghosts = {}
        for group_id, group in config.link_groups.items():
            for agent_id in group.all_agents():
                assignment = config.workers.get(agent_id)
                if assignment and assignment.worker_type != "rl":
                    ghosts[agent_id] = group_id

        # No link groups => no ghosts (each agent is independent)
        assert ghosts == {}
