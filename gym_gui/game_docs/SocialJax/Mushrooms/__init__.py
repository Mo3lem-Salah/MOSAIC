"""Documentation for SocialJax Mushrooms environment."""
from __future__ import annotations

MUSHROOMS_HTML = """
<h2>SocialJax: Mushrooms</h2>

<p><strong>Environment ID:</strong> <code>mushrooms</code></p>

<h3>Overview</h3>
<p>
Agents forage for mushrooms in a grid world. Some mushrooms are nutritious (positive reward)
and some are toxic (negative reward to neighbouring agents who consume them). This tests
whether agents can develop norms about which mushrooms to eat and whether they account for
the <strong>externalities</strong> of their foraging choices on others.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2-4</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Externalities / Mixed Incentives</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Backend</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pure JAX (GPU/TPU/CPU)</td></tr>
</table>

<h3>Mushroom Types &amp; Reward</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Type</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Colour</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Self Reward</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Others Reward</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Digestion Freeze</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Regrow Prob</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Red</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Red</td>
        <td style="border: 1px solid #ddd; padding: 8px;">+1.0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">10 steps</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0.25</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Green</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Green</td>
        <td style="border: 1px solid #ddd; padding: 8px;">+2.0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">15 steps</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0.40</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Blue</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Blue</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">+3.0 to all others</td>
        <td style="border: 1px solid #ddd; padding: 8px;">20 steps</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0.60</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Orange (toxic)</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Orange</td>
        <td style="border: 1px solid #ddd; padding: 8px;">-0.2</td>
        <td style="border: 1px solid #ddd; padding: 8px;">-0.2 to all others</td>
        <td style="border: 1px solid #ddd; padding: 8px;">0 (instant)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1.0 (always)</td>
    </tr>
</table>

<h3>Digestion Mechanic</h3>
<p>
After eating a mushroom, the agent enters a <strong>frozen digestion state</strong> for the specified number of steps.
During digestion, the agent cannot eat another mushroom. Higher-value mushrooms freeze the agent longer.
Orange mushrooms are toxic: -0.2 to the eater and -0.2 to all other agents. They regrow instantly (probability 1.0),
so they persist as a trap for the entire episode.
</p>

<h3>Social Dilemma</h3>
<p>
<strong>Externality game:</strong> Red/green mushrooms are private goods (selfish reward). Blue mushrooms are pure public goods &mdash;
the eater gets nothing but all others get +3. Orange mushrooms are pure negative externalities &mdash; eating hurts everyone.
The social dilemma is whether agents learn to eat blue (altruistic) and avoid orange (harmful), or just maximise private reward via green mushrooms.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a></i></p>
"""
