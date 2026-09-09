"""Shared HTML fragments for Jidi Olympics environment documentation."""

OLYMPICS_OVERVIEW_HTML = """
<h4>Jidi Olympics</h4>
<p>
The Jidi Olympics platform provides physics-based competitive multi-agent
environments. Agents control elastic ball entities that interact through
realistic collision dynamics, energy management, and continuous force control.
</p>
<p>
<b>Source:</b> <code>3rd_party/environments/Competition_Olympics-Wrestling/</code>
</p>
"""

OLYMPICS_ACTIONS_HTML = """
<h4>Action Space (Continuous, 2 dimensions)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Dimension</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Range</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">Force</td>
        <td style="border: 1px solid #ddd; padding: 6px;">[-100, 200]</td>
        <td style="border: 1px solid #ddd; padding: 6px;">Driving force magnitude. Negative = reverse thrust. Energy cost is proportional to force * velocity.</td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;">Angle</td>
        <td style="border: 1px solid #ddd; padding: 6px;">[-30, 30] degrees</td>
        <td style="border: 1px solid #ddd; padding: 6px;">Steering angle change per step. Accumulates across steps (heading = sum of all angle actions).</td>
    </tr>
</table>
<h4>Keyboard Presets (Human Control)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #d4edda;">
        <th style="border: 1px solid #ddd; padding: 6px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Force / Angle</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">W</td><td style="border: 1px solid #ddd; padding: 6px;">Forward</td><td style="border: 1px solid #ddd; padding: 6px;">150 / 0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">S</td><td style="border: 1px solid #ddd; padding: 6px;">Backward</td><td style="border: 1px solid #ddd; padding: 6px;">-80 / 0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">A</td><td style="border: 1px solid #ddd; padding: 6px;">Turn Left</td><td style="border: 1px solid #ddd; padding: 6px;">0 / -20</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">D</td><td style="border: 1px solid #ddd; padding: 6px;">Turn Right</td><td style="border: 1px solid #ddd; padding: 6px;">0 / +20</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Q</td><td style="border: 1px solid #ddd; padding: 6px;">Forward-Left</td><td style="border: 1px solid #ddd; padding: 6px;">150 / -15</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">E</td><td style="border: 1px solid #ddd; padding: 6px;">Forward-Right</td><td style="border: 1px solid #ddd; padding: 6px;">150 / +15</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Shift</td><td style="border: 1px solid #ddd; padding: 6px;">Sprint</td><td style="border: 1px solid #ddd; padding: 6px;">200 / 0</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Space</td><td style="border: 1px solid #ddd; padding: 6px;">Idle</td><td style="border: 1px solid #ddd; padding: 6px;">0 / 0</td></tr>
</table>
"""

OLYMPICS_OBS_HTML = """
<h4>Observation Space (40x40 egocentric grid)</h4>
<p>Each agent receives a <b>40x40 integer grid</b> representing its agent-relative
field of view. The grid is rotated to the agent's heading direction, so the
agent always "looks forward." Each cell contains a color index:</p>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Index</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Color</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Meaning</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">0</td><td style="border: 1px solid #ddd; padding: 6px;">Light green</td><td style="border: 1px solid #ddd; padding: 6px;">Empty arena floor</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">1</td><td style="border: 1px solid #ddd; padding: 6px;">Green</td><td style="border: 1px solid #ddd; padding: 6px;">Arena boundary (fall off = lose)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">4</td><td style="border: 1px solid #ddd; padding: 6px;">Grey</td><td style="border: 1px solid #ddd; padding: 6px;">Cross/division lines</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">5</td><td style="border: 1px solid #ddd; padding: 6px;">Purple</td><td style="border: 1px solid #ddd; padding: 6px;">Self (own agent body)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">7</td><td style="border: 1px solid #ddd; padding: 6px;">Red/Blue</td><td style="border: 1px solid #ddd; padding: 6px;">Opponent agent body</td></tr>
</table>
<p><b>Visibility:</b> 200 units radius, 5 units per cell = 40x40 grid.</p>
"""

OLYMPICS_PHYSICS_HTML = """
<h4>Physics and Energy System</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Parameter</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Value</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Description</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Time step (tau)</td><td style="border: 1px solid #ddd; padding: 6px;">0.1</td><td style="border: 1px solid #ddd; padding: 6px;">Physics simulation timestep</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Speed cap</td><td style="border: 1px solid #ddd; padding: 6px;">100</td><td style="border: 1px solid #ddd; padding: 6px;">Maximum velocity magnitude</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Energy cap</td><td style="border: 1px solid #ddd; padding: 6px;">1000</td><td style="border: 1px solid #ddd; padding: 6px;">Maximum energy per agent</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Energy recovery</td><td style="border: 1px solid #ddd; padding: 6px;">200/s</td><td style="border: 1px solid #ddd; padding: 6px;">Fixed rate energy regeneration</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Energy cost</td><td style="border: 1px solid #ddd; padding: 6px;">force * velocity / 50</td><td style="border: 1px solid #ddd; padding: 6px;">Per-step consumption</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">Restitution</td><td style="border: 1px solid #ddd; padding: 6px;">1.0 (elastic)</td><td style="border: 1px solid #ddd; padding: 6px;">Wall and agent-agent collisions</td></tr>
</table>
<p><b>Fatigue:</b> When energy reaches 0, the agent cannot apply force until
energy recovers. Strategic energy management is essential.</p>
"""
