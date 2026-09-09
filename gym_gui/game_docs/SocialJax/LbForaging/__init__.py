"""Documentation for SocialJax Level-Based Foraging environment."""
from __future__ import annotations

LB_FORAGING_HTML = """
<h2>SocialJax: Level-Based Foraging</h2>

<p><strong>Environment ID:</strong> <code>lb_foraging</code></p>

<h3>Overview</h3>
<p>
Agents and food items have <strong>levels</strong>. An agent can only pick up food whose
level is less than or equal to the sum of all agents simultaneously attempting the pickup.
This means high-level food requires cooperative simultaneous action. Agents must identify
partners, agree on targets, and converge at the right time.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2-4</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Coordination / Cooperative Foraging</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Stepping</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Parallel (simultaneous)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Backend</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Pure JAX (GPU/TPU/CPU)</td></tr>
</table>

<h3>Reward Mechanics</h3>
<p>
Each food item has a <strong>level</strong>. An agent can only pick up food if the sum of levels of all agents
currently adjacent to (and attempting to pick up) that food meets or exceeds the food's level.
</p>

<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Scenario</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Reward</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Formula</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Successful pickup</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>food_level &times; agent_level / total_agent_level</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Each agent's share is proportional to its level relative to the total level of all participating agents</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Insufficient levels</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">0</td>
        <td style="border: 1px solid #ddd; padding: 8px;">If total adjacent agent levels &lt; food level, no pickup occurs</td>
    </tr>
</table>

<h3>Example</h3>
<p>
Food level 4, Agent A (level 2) and Agent B (level 3) both attempt pickup: total = 5 &ge; 4, success.<br/>
Agent A gets: 4 &times; 2/5 = <strong>1.6</strong>. Agent B gets: 4 &times; 3/5 = <strong>2.4</strong>.<br/>
If only Agent B (level 3) attempts: 3 &lt; 4, no pickup. Cooperation required.
</p>

<h3>Social Dilemma</h3>
<p>
<strong>Pure coordination:</strong> No conflict between agents &mdash; all rewards are positive and picking up food benefits everyone involved.
The challenge is spatiotemporal coordination: agents must converge on the same food item at the same time.
Higher-level food requires more agents, making coordination exponentially harder.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a></i></p>
"""
