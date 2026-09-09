"""Documentation for SocialJax Commons Harvest (open) environment."""
from __future__ import annotations

HARVEST_COMMON_OPEN_HTML = """
<h2>SocialJax: Commons Harvest (Open)</h2>

<p><strong>Environment ID:</strong> <code>harvest_common_open</code></p>

<h3>Overview</h3>
<p>
Agents harvest apples from a shared orchard. Apples regrow at a rate proportional to the
density of surrounding apples. Over-harvesting depletes the resource and collapses regrowth
for all agents &mdash; the classic <strong>tragedy of the commons</strong>. Sustainable
collective behaviour requires agents to self-regulate their harvest rate.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2-6</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Commons / Tragedy of the Commons</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Backend</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pure JAX (GPU/TPU/CPU)</td></tr>
</table>

<h3>Reward Mechanics</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Reward</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Mechanic</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Eat apple</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">+1.0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Agent occupies cell with apple; apple consumed</td>
    </tr>
</table>

<h3>Apple Regrowth (Density-Dependent)</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Parameter</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Regrowth neighbourhood</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Radius 2 (L-infinity)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Counts apples within a 5x5 square centred on the empty cell</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Regrowth probability</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Proportional to density</td>
        <td style="border: 1px solid #ddd; padding: 8px;">More nearby apples &rarr; higher regrowth chance (positive feedback)</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Depletion threshold</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Near-zero density</td>
        <td style="border: 1px solid #ddd; padding: 8px;">When too few apples remain nearby, regrowth probability collapses &rarr; resource extinction</td>
    </tr>
</table>

<h3>Social Dilemma</h3>
<p>
<strong>Tragedy of the commons:</strong> Each apple harvested is +1 to the individual, but reduces the neighbourhood density that drives regrowth for everyone.
Sustainable harvesting requires agents to leave apples for regrowth. If all agents harvest greedily, the orchard collapses and total reward drops to zero.
The open variant has no spatial barriers &mdash; agents can access the entire commons.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a></i></p>
"""
