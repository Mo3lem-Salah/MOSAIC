"""Documentation for SocialJax Coin Game environment."""
from __future__ import annotations

COIN_GAME_HTML = """
<h2>SocialJax: Coin Game</h2>

<p><strong>Environment ID:</strong> <code>coin_game</code></p>

<h3>Overview</h3>
<p>
Two agents collect coins of two colours. Collecting your own colour gives +1; collecting the
other agent's colour gives +1 to you but -2 to them. This creates a tension between selfish
coin-grabbing and altruistic restraint: the socially optimal policy is for each agent to
collect only their own colour.
</p>

<h3>Visual Guide</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 8px;">
    <tr style="background-color:#f2f2f2;">
        <th style="border:1px solid #ddd;padding:6px;">Agent</th>
        <th style="border:1px solid #ddd;padding:6px;">Body Color</th>
        <th style="border:1px solid #ddd;padding:6px;">Their Coins</th>
        <th style="border:1px solid #ddd;padding:6px;">Rule</th>
    </tr>
    <tr>
        <td style="border:1px solid #ddd;padding:6px;">Agent 0</td>
        <td style="border:1px solid #ddd;padding:6px;"><span style="color:#d62728;font-weight:bold;">Red triangle</span></td>
        <td style="border:1px solid #ddd;padding:6px;"><span style="color:#d62728;font-weight:bold;">Red circles</span></td>
        <td style="border:1px solid #ddd;padding:6px;">Collect red circles (+1). Avoid green circles.</td>
    </tr>
    <tr>
        <td style="border:1px solid #ddd;padding:6px;">Agent 1</td>
        <td style="border:1px solid #ddd;padding:6px;"><span style="color:#17becf;font-weight:bold;">Blue/cyan triangle</span></td>
        <td style="border:1px solid #ddd;padding:6px;"><span style="color:#2ca02c;font-weight:bold;">Green circles</span></td>
        <td style="border:1px solid #ddd;padding:6px;">Collect green circles (+1). Avoid red circles.</td>
    </tr>
</table>
<p><em>Note: Agent 1 has a blue body but green coins. Coin colors are always red and green; agent body colors follow an HSV color wheel (red at hue=0, cyan at hue=0.5 for 2 agents).</em></p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Mixed Incentives / Social Dilemma</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Backend</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pure JAX (GPU/TPU/CPU)</td></tr>
</table>

<h3>Payoff Matrix</h3>
<p>
The reward is determined by the <strong>payoff matrix</strong>. When an agent collects a coin:
</p>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Event</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Collector Gets</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Coin Owner Gets</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Own colour collected</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">+1.0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">N/A (same agent)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Other's colour collected</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">+1.0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">-2.0 (externality)</td>
    </tr>
</table>
<p>
<code>payoff_matrix = [[1, 1], [-2, -2]]</code> &mdash; row 0 = collector reward (+1 always), row 1 = externality to coin owner (-2 when other-colour collected).
The social dilemma: grabbing any coin is +1 to you, but costs the other agent -2 if it was their colour.
</p>

<h3>Social Dilemma</h3>
<p>
The <strong>socially optimal</strong> strategy is for each agent to collect only their own colour: +1 each per coin, no negative externalities.
The <strong>selfish</strong> strategy grabs everything: +1 per coin to collector, but -2 penalty to the owner of wrong-colour coins.
If both agents grab indiscriminately, the net expected payoff is lower than mutual restraint.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a></i></p>
"""
