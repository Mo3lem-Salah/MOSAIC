"""Documentation for SocialJax Territory (open) environment."""
from __future__ import annotations

TERRITORY_OPEN_HTML = """
<h2>SocialJax: Territory (Open)</h2>

<p><strong>Environment ID:</strong> <code>territory_open</code></p>

<h3>Overview</h3>
<p>
Two agents claim and defend spatial territory on an open grid. Agents paint cells with their
colour by moving over them. Conflict emerges at contested borders. The total reward is
proportional to the fraction of territory controlled at episode end.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Competitive / Territory Control</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Backend</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pure JAX (GPU/TPU/CPU)</td></tr>
</table>

<h3>Reward Mechanics</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Effect</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Details</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Move</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Paint cell</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Agent paints the cell it moves to with its colour. Overwrites opponent's paint.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Zap</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Destroy forward</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Destroys walls and can stun opponent agents in the beam path. Costs a step of movement.</td>
    </tr>
</table>

<h3>Territory Scoring</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Mechanism</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Details</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Paint drying</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Paint takes <strong>25 steps</strong> to dry. Only dry paint counts as territory. Fresh paint can still be overwritten by the opponent.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Episode-end scoring</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">At episode end, each agent's reward = fraction of total cells painted in their colour (dry or fresh). Zero-sum: both fractions sum to 1.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Walls</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Walls are obstacles that can be destroyed with zap. Strategic wall destruction opens new territory access.</td>
    </tr>
</table>

<h3>Social Dilemma</h3>
<p>
<strong>Zero-sum competition:</strong> Territory is purely competitive &mdash; one agent's gain is the other's loss.
The drying delay (25 steps) creates a strategic tension: paint near the border is vulnerable to overwriting before it dries.
The zap mechanic adds direct adversarial interaction beyond simple space coverage.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a></i></p>
"""
