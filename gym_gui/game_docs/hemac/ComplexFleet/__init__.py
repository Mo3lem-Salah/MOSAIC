"""Documentation for HeMAC Complex Fleet environments."""
from __future__ import annotations

_COMPLEX_FLEET_SCENARIOS = {
    "3q1o1p": {"quadcopters": 3, "observers": 1, "provisioners": 1},
    "5q2o1p": {"quadcopters": 5, "observers": 2, "provisioners": 1},
}


def get_complex_fleet_html(env_id: str) -> str:
    """Generate Complex Fleet HTML documentation for a specific variant."""
    for key, comp in _COMPLEX_FLEET_SCENARIOS.items():
        if key in env_id:
            composition = comp
            break
    else:
        composition = {"quadcopters": "?", "observers": "?", "provisioners": "?"}

    total = composition["quadcopters"] + composition["observers"] + composition["provisioners"]

    return f"""
<h2>{env_id}</h2>

<p style="background-color: #fff3e0; padding: 8px; border-radius: 4px; margin-bottom: 10px;">
<strong>API:</strong> PettingZoo Parallel — simultaneous stepping.
<a href="https://github.com/ThalesGroup/HeMAC" target="_blank">GitHub</a>
&nbsp;|&nbsp; ECAI 2025 &mdash; ThalesGroup
</p>

<p>Complex Fleet challenge: the full heterogeneous benchmark with all three
agent types. Provisioners (ground vehicles) relay observer intel and
support drones, adding a communication-routing layer on top of the aerial
coordination task. {total} agents
({composition['quadcopters']}Q + {composition['observers']}O + {composition['provisioners']}P).</p>

<h4>Available Variants</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Environment ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Q</th>
        <th style="border: 1px solid #ddd; padding: 8px;">O</th>
        <th style="border: 1px solid #ddd; padding: 8px;">P</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Total</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Max Cycles</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-complex-fleet-3q1o1p-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">3</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">5</td>
        <td style="border: 1px solid #ddd; padding: 8px;">900</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>hemac-complex-fleet-5q2o1p-v0</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">5</td>
        <td style="border: 1px solid #ddd; padding: 8px;">2</td>
        <td style="border: 1px solid #ddd; padding: 8px;">1</td>
        <td style="border: 1px solid #ddd; padding: 8px;">8</td>
        <td style="border: 1px solid #ddd; padding: 8px;">900</td>
    </tr>
</table>

<h4>Agent Types &amp; Spaces</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Agent</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Obs Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Action Space</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Role</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Quadcopter (Q)</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(-10000, 10000, (15,))</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(5)</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Low-altitude interceptor. Max speed 12. Energy-constrained.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Observer (O)</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(-10000, 10000, (11,))</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(5)</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">High-altitude surveillance. ForwardFacingCamera (hfov=30&deg;, range 200).</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>Provisioner (P)</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Box(-10000, 10000, (8,))</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;"><code>Discrete(5)</code></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Road-constrained ground vehicle. Speed 8. Relays observer intel to drones.</td>
    </tr>
</table>

<h4>Action Space: Discrete(5)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #e3f2fd;">
        <th style="border: 1px solid #ddd; padding: 8px;">ID</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Drone / Observer</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Provisioner (ground)</th>
        <th style="border: 1px solid #ddd; padding: 8px;">MOSAIC key</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>0</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">NOOP / recharge</td>
        <td style="border: 1px solid #ddd; padding: 8px;">STAY</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><kbd>Space</kbd></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>1</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Fly SE &nbsp;(+vx, +vy)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">DRIVE_EAST</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><kbd>D</kbd> / <kbd>&rarr;</kbd></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>2</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Fly NE &nbsp;(+vx, &minus;vy)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">DRIVE_NORTH</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><kbd>W</kbd> / <kbd>&uarr;</kbd></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>3</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Fly SW &nbsp;(&minus;vx, +vy)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">DRIVE_WEST</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><kbd>S</kbd> / <kbd>&darr;</kbd></td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 8px;"><strong>4</strong></td>
        <td style="border: 1px solid #ddd; padding: 8px;">Fly NW &nbsp;(&minus;vx, &minus;vy)</td>
        <td style="border: 1px solid #ddd; padding: 8px;">DRIVE_SOUTH</td>
        <td style="border: 1px solid #ddd; padding: 8px;"><kbd>A</kbd> / <kbd>&larr;</kbd></td>
    </tr>
</table>
<p style="color: #888; font-size: 0.9em;">Drone movements are diagonal. W/D are semantically consistent for provisioners
(W=NORTH, D=EAST) but A/S are swapped (A=SOUTH, S=WEST) because the drone action
integers don&apos;t align with cardinal directions for all four keys. In Human Only
mode the same action goes to every agent simultaneously.</p>

<h4>Scenario Parameters</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 8px;">Parameter</th>
        <th style="border: 1px solid #ddd; padding: 8px;">Value</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Obstacles</td><td style="border: 1px solid #ddd; padding: 8px;">2&ndash;5 random per episode</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">POI speed</td><td style="border: 1px solid #ddd; padding: 8px;">3.0, random spawn</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Drone max speed</td><td style="border: 1px solid #ddd; padding: 8px;">12</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Provisioner speed</td><td style="border: 1px solid #ddd; padding: 8px;">8 (road-constrained)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 8px;">Max cycles</td><td style="border: 1px solid #ddd; padding: 8px;">900 (configurable)</td></tr>
</table>

<h4>References</h4>
<ul>
    <li><a href="https://github.com/ThalesGroup/HeMAC" target="_blank">HeMAC GitHub (ThalesGroup)</a></li>
    <li>Dansereau et al. (2025). "The Heterogeneous Multi-Agent Challenge." ECAI 2025. arXiv:2509.19512</li>
</ul>
"""


HEMAC_COMPLEX_FLEET_HTML = get_complex_fleet_html("hemac-complex-fleet-3q1o1p-v0")

__all__ = ["HEMAC_COMPLEX_FLEET_HTML", "get_complex_fleet_html"]
