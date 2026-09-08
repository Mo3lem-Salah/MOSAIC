"""Documentation for SocialJax Prisoner's Dilemma Arena environment."""
from __future__ import annotations

PD_ARENA_HTML = """
<h2>SocialJax: Prisoner's Dilemma Arena</h2>

<p><strong>Environment ID:</strong> <code>pd_arena</code></p>

<h3>Overview</h3>
<p>
Two agents play an iterated Prisoner's Dilemma embedded in a spatial grid world. Agents can
cooperate (C) or defect (D). The standard payoff matrix applies: mutual cooperation yields
moderate reward for both; defecting against a cooperator yields a high reward for the
defector and a penalty for the cooperator; mutual defection is worst for both.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Game Theory / Social Dilemma</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Backend</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pure JAX (GPU/TPU/CPU)</td></tr>
</table>

<h3>Payoff Matrix (Prisoner's Dilemma)</h3>
<p>
Actions: <strong>Cooperate (C)</strong> or <strong>Defect (D)</strong>. Each cell interaction applies the payoff:
</p>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Agent 0 \ Agent 1</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Cooperate</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Defect</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Cooperate</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">(+3, +3)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">(-1, +5)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Defect</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">(+5, -1)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">(+1, +1)</td>
    </tr>
</table>
<p>
<code>payoff_matrix = [[[3, -1], [5, 1]], [[3, 5], [-1, 1]]]</code> &mdash; first index = agent 0's action (C/D), second = agent 1's action.
</p>

<h3>Spatial Embedding</h3>
<p>
Unlike a tabletop PD, agents move on a grid and must be <strong>adjacent</strong> to interact.
The iterated nature (multiple steps per episode) allows strategies like tit-for-tat, forgiveness, and reputation-building.
</p>

<h3>Social Dilemma</h3>
<p>
<strong>Classic prisoner's dilemma:</strong> Defecting dominates cooperating in a single round (5 > 3 and 1 > -1).
But mutual defection (+1 each) is worse than mutual cooperation (+3 each).
In the spatial iterated version, cooperative clusters can survive if they can isolate defectors.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a></i></p>
"""
