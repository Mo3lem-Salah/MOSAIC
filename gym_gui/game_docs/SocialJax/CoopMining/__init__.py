"""Documentation for SocialJax Cooperative Mining environment."""
from __future__ import annotations

COOP_MINING_HTML = """
<h2>SocialJax: Cooperative Mining</h2>

<p><strong>Environment ID:</strong> <code>coop_mining</code></p>

<h3>Overview</h3>
<p>
Agents must mine iron and gold ore from a shared 28x28 grid world. Iron can be mined by a
single agent, but gold requires coordinated mining by exactly 2 agents within a 3-step window.
This creates a fundamental coordination problem: agents must recognise when a gold node is
present and converge on it within the time window to unlock the higher-value reward.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2-6 (default: 6)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Grid Size</strong></td><td style="border: 1px solid #ddd; padding: 8px;">28 x 28 tiles</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Episode Length</strong></td><td style="border: 1px solid #ddd; padding: 8px;">1000 steps (num_inner_steps)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Fully Cooperative MARL</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>MDP Type</strong></td><td style="border: 1px solid #ddd; padding: 8px;">PO-MDP (Partially Observable)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Observation</strong></td><td style="border: 1px solid #ddd; padding: 8px;">11x11 egocentric window, 12 channels (1452-dim flat)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Action Space</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Discrete(8): turn-L, turn-R, step-L, step-R, forward, backward, stay, mine</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Primary Metric</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Cumulative shared reward per episode (not win/loss)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Backend</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pure JAX (GPU/TPU/CPU)</td></tr>
</table>

<h3>Reward Mechanics</h3>
<p>
There are two types of ore with different mining requirements and reward values.
All rewards are shared: when <code>shared_rewards=True</code> (default), every agent receives
the total team reward each step.
</p>

<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Ore Type</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Reward</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Miners Required</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Time Window</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Regrowth Prob</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Mechanic</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Iron Ore</strong> (brown)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">+1.0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1 (solo)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">N/A</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0.0004/step</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Agent mines forward up to 3 tiles. First ore hit is collected. Becomes <code>ore_wait</code> after mining.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Gold Ore</strong> (yellow)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">+8.0 per agent</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Exactly 2</td>
        <td style="border: 1px solid #ddd; padding: 8px;">3 steps</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0.00016/step</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1st agent mines: becomes <code>gold_partial</code> (brighter yellow). 2nd agent mines within window: both get +8, ore becomes <code>ore_wait</code>. If &gt;2 agents mine: ore reverts, no reward. If window expires: reverts to <code>gold_ore</code>.</td>
    </tr>
</table>

<h3>Gold Mining State Machine</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">State</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Trigger</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Next State</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Reward</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>gold_ore</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">1st agent mines</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>gold_partial</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">0 (waiting for partner)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>gold_partial</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">2nd agent mines within 3 steps</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>ore_wait</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">+8.0 to each agent</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>gold_partial</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">&gt;2 agents mine (overcrowding)</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>gold_ore</code> (revert)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0 (coordination failure)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>gold_partial</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3-step window expires</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>gold_ore</code> (revert)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0 (timeout)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>ore_wait</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Stochastic regrowth</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>iron_ore</code> or <code>gold_ore</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">N/A</td>
    </tr>
</table>

<h3>Observation (PO-MDP)</h3>
<p>
Each agent sees only a local <strong>11x11 egocentric window</strong> (forward=9, backward=1, left=5, right=5)
rotated to match its orientation. The full grid is 28x28, so agents cannot see most of the map.
This partial observability is what makes coordination challenging &mdash; agents must learn to
approach gold nodes they can see and trust that partners will do the same.
</p>

<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Channel</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">0-5</td><td style="border: 1px solid #ddd; padding: 8px;">Item one-hot: wall, ore_wait, spawn_point, iron_ore, gold_ore, gold_partial</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">6</td><td style="border: 1px solid #ddd; padding: 8px;">"This is me" flag</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">7</td><td style="border: 1px solid #ddd; padding: 8px;">"Other agent" flag</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">8-11</td><td style="border: 1px solid #ddd; padding: 8px;">Agent orientation one-hot (N/E/S/W relative to viewer)</td></tr>
</table>

<h3>Social Dilemma</h3>
<p>
The selfish strategy is to mine iron independently (+1 each); the cooperative strategy is to
find a partner and mine gold together (+8 each). Gold requires exactly 2 agents &mdash; too few
means no reward, too many means the ore reverts. This "overcrowding penalty" creates a
coordination game on top of the cooperation requirement.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a> (ICLR 2026)</i></p>
"""
