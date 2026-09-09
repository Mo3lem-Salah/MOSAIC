"""Documentation for SocialJax Gift Exchange environment."""
from __future__ import annotations

GIFT_HTML = """
<h2>SocialJax: Gift Exchange</h2>

<p><strong>Environment ID:</strong> <code>gift</code></p>

<h3>Overview</h3>
<p>
Agents can give resources to other agents at a personal cost. Giving yields a larger benefit
to the receiver than the cost to the giver. This tests whether agents develop <strong>reciprocity
norms</strong> and trust: the socially optimal strategy is mutual gift-giving, but it requires
agents to learn to give first without guarantee of return.
</p>

<h3>Specifications</h3>
<table style="border-collapse: collapse; width: 100%; margin-top: 10px;">
    <tr>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Property</th>
        <th style="border: 1px solid #ddd; padding: 8px; background-color: #f2f2f2;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Agents</strong></td><td style="border: 1px solid #ddd; padding: 8px;">2-4</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;"><strong>Category</strong></td><td style="border: 1px solid #ddd; padding: 8px;">Reciprocity / Trust / Cooperation</td></tr>
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
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Collect token</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">+0 (held)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Agent picks up token from grid; token added to inventory. Tokens spawn with probability <code>p = 0.0002</code> per empty cell per step.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Gift (give)</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">&times;3 refinement</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Agent gives all held tokens to a neighbouring agent. Tokens are <strong>tripled</strong>: giver loses <code>n</code>, receiver gets <code>3n</code>. The "refinement" multiplier rewards prosocial behaviour.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Consume</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">+1.0 per token</td>
        <td style="border: 1px solid #ddd; padding: 8px;">Agent converts all held tokens to reward (+1 each). Tokens are destroyed. No refinement &mdash; consuming yields less total value than gifting in a cooperative loop.</td>
    </tr>
</table>

<h3>Refinement Loop</h3>
<p>
The <code>refine_coin = 2</code> parameter means gifting multiplies tokens by 3 (1 + refine_coin).
The optimal cooperative loop: Agent A collects tokens &rarr; gifts to Agent B (tokens &times;3) &rarr; B gifts back (tokens &times;3 again) &rarr; repeat.
Each gift-giving cycle triples the total tokens. Eventually, consuming yields far more reward than any agent could collect alone.
</p>

<h3>Social Dilemma</h3>
<p>
<strong>Trust game:</strong> Gifting has no immediate personal reward &mdash; the giver must trust the receiver to reciprocate.
A selfish agent collects tokens and consumes immediately (+1 each). A cooperative pair can triple tokens through mutual gifting,
then consume for 3&times;+ the reward. But defection (receiving gifts without returning) is always tempting.
</p>

<p><i>Source: SocialJax &mdash; <a href="https://arxiv.org/abs/2503.14576">arxiv:2503.14576</a></i></p>
"""
