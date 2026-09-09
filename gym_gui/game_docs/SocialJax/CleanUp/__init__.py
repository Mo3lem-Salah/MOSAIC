"""Documentation for SocialJax Clean Up environment."""
from __future__ import annotations

CLEAN_UP_HTML = """
<h2>SocialJax: Clean Up</h2>

<p><strong>Environment ID:</strong> <code>clean_up</code></p>

<h3>Overview</h3>
<p>
Agents can collect apples or clean pollution from a river. Apples only regrow when the river
is clean enough. Cleaning is costly for the cleaning agent but benefits all apple collectors.
This is a classic <strong>public goods</strong> social dilemma: rational selfish agents
free-ride on others' cleaning effort.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2-5</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Public Goods / Altruism</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Backend</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pure JAX (GPU/TPU/CPU)</td></tr>
</table>

<h3>Reward Mechanics</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Action</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Reward</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Condition</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Effect</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Eat apple</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">+1.0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Apple present on agent's cell</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Apple consumed, removed from grid</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Clean (beam)</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">0 (costly)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cleaning beam fires forward</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Removes pollution from cells in beam path; agent forfeits apple-collecting that step</td>
    </tr>
</table>

<h3>Pollution &amp; Apple Regrowth</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Mechanism</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Details</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Apple regrowth</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Empty apple cells regrow with probability proportional to nearby apple density (radius 2), <em>only if</em> local pollution is below <code>thresholdDepletion = 0.4</code></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Pollution accumulation</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Pollution increases over time in the river area; when it exceeds the threshold, apple regrowth stops entirely</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Cleaning effect</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Cleaning beam reduces pollution in affected cells, restoring apple regrowth potential for all agents</td>
    </tr>
</table>

<h3>Social Dilemma</h3>
<p>
<strong>Free-rider problem:</strong> Cleaning costs an agent a step (no apple collection that turn) but benefits everyone by restoring apple regrowth.
Rational selfish agents prefer to collect apples while others clean. If all agents free-ride, pollution accumulates, apples stop growing, and everyone earns zero.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a></i></p>
"""
