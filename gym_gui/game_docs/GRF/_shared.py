"""Shared HTML fragments for Google Research Football (GRF) documentation."""

GRF_OVERVIEW_HTML = """
<h4>Google Research Football (GRF)</h4>
<p>
GRF is an RL environment based on the popular game of football (soccer),
developed by the Google Brain team. It simulates a full football game with
realistic physics, 22 players, offsides, fouls, corner kicks, penalty kicks,
and other standard rules.
</p>
<p>
<b>Paper:</b> Kurach et al. (2020). "Google Research Football: A Novel
Reinforcement Learning Environment"<br/>
<b>Source:</b> <code>3rd_party/environments/football/</code>
</p>
"""

GRF_ACTIONS_HTML = """
<h4>Action Space (Discrete, 19 actions)</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Index</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Action</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Key</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Notes</th>
    </tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">0</td><td style="border: 1px solid #ddd; padding: 6px;">idle</td><td style="border: 1px solid #ddd; padding: 6px;">0</td><td style="border: 1px solid #ddd; padding: 6px;">No action</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">1</td><td style="border: 1px solid #ddd; padding: 6px;">left</td><td style="border: 1px solid #ddd; padding: 6px;">A</td><td style="border: 1px solid #ddd; padding: 6px;">Move left</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">2</td><td style="border: 1px solid #ddd; padding: 6px;">top_left</td><td style="border: 1px solid #ddd; padding: 6px;">Q</td><td style="border: 1px solid #ddd; padding: 6px;">Move diagonally</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">3</td><td style="border: 1px solid #ddd; padding: 6px;">top</td><td style="border: 1px solid #ddd; padding: 6px;">W</td><td style="border: 1px solid #ddd; padding: 6px;">Move up</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">4</td><td style="border: 1px solid #ddd; padding: 6px;">top_right</td><td style="border: 1px solid #ddd; padding: 6px;">E</td><td style="border: 1px solid #ddd; padding: 6px;">Move diagonally</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">5</td><td style="border: 1px solid #ddd; padding: 6px;">right</td><td style="border: 1px solid #ddd; padding: 6px;">D</td><td style="border: 1px solid #ddd; padding: 6px;">Move right</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">6</td><td style="border: 1px solid #ddd; padding: 6px;">bottom_right</td><td style="border: 1px solid #ddd; padding: 6px;">C</td><td style="border: 1px solid #ddd; padding: 6px;">Move diagonally</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">7</td><td style="border: 1px solid #ddd; padding: 6px;">bottom</td><td style="border: 1px solid #ddd; padding: 6px;">S</td><td style="border: 1px solid #ddd; padding: 6px;">Move down</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">8</td><td style="border: 1px solid #ddd; padding: 6px;">bottom_left</td><td style="border: 1px solid #ddd; padding: 6px;">Z</td><td style="border: 1px solid #ddd; padding: 6px;">Move diagonally</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">9</td><td style="border: 1px solid #ddd; padding: 6px;">long_pass</td><td style="border: 1px solid #ddd; padding: 6px;">J</td><td style="border: 1px solid #ddd; padding: 6px;">Long through ball</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">10</td><td style="border: 1px solid #ddd; padding: 6px;">high_pass</td><td style="border: 1px solid #ddd; padding: 6px;">K</td><td style="border: 1px solid #ddd; padding: 6px;">Lofted pass</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">11</td><td style="border: 1px solid #ddd; padding: 6px;">short_pass</td><td style="border: 1px solid #ddd; padding: 6px;">L</td><td style="border: 1px solid #ddd; padding: 6px;">Ground pass</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">12</td><td style="border: 1px solid #ddd; padding: 6px;">shot</td><td style="border: 1px solid #ddd; padding: 6px;">Space</td><td style="border: 1px solid #ddd; padding: 6px;">Shoot at goal</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">13</td><td style="border: 1px solid #ddd; padding: 6px;">sprint</td><td style="border: 1px solid #ddd; padding: 6px;">Shift</td><td style="border: 1px solid #ddd; padding: 6px;">Toggle sprint mode</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">14</td><td style="border: 1px solid #ddd; padding: 6px;">release_direction</td><td style="border: 1px solid #ddd; padding: 6px;">X</td><td style="border: 1px solid #ddd; padding: 6px;">Stop moving</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">15</td><td style="border: 1px solid #ddd; padding: 6px;">release_sprint</td><td style="border: 1px solid #ddd; padding: 6px;">V</td><td style="border: 1px solid #ddd; padding: 6px;">Stop sprinting</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">16</td><td style="border: 1px solid #ddd; padding: 6px;">sliding</td><td style="border: 1px solid #ddd; padding: 6px;">T</td><td style="border: 1px solid #ddd; padding: 6px;">Sliding tackle (defense)</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">17</td><td style="border: 1px solid #ddd; padding: 6px;">dribble</td><td style="border: 1px solid #ddd; padding: 6px;">R</td><td style="border: 1px solid #ddd; padding: 6px;">Toggle dribble mode</td></tr>
    <tr><td style="border: 1px solid #ddd; padding: 6px;">18</td><td style="border: 1px solid #ddd; padding: 6px;">release_dribble</td><td style="border: 1px solid #ddd; padding: 6px;">F</td><td style="border: 1px solid #ddd; padding: 6px;">Stop dribbling</td></tr>
</table>
<p><b>Sticky actions:</b> sprint and dribble are toggles that persist across
steps until explicitly released.</p>
"""

GRF_OBS_HTML = """
<h4>Observation Space</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Representation</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Shape</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><code>simple115v2</code> (default)</td>
        <td style="border: 1px solid #ddd; padding: 6px;">(115,)</td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            Fixed-length vector encoding: ball position/direction (6),
            left team positions/directions (22), right team positions/directions (22),
            active player one-hot (11), sticky actions (10), game mode one-hot (7),
            ball ownership, score delta
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><code>extracted</code></td>
        <td style="border: 1px solid #ddd; padding: 6px;">(72, 96, 4)</td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            "Super minimap": 4-channel image encoding player positions,
            ball, and active player on a 72x96 grid
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><code>pixels</code></td>
        <td style="border: 1px solid #ddd; padding: 6px;">(72, 96, 3)</td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            Downscaled RGB game frame (requires <code>render=True</code>)
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><code>raw</code></td>
        <td style="border: 1px solid #ddd; padding: 6px;">dict</td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            Full game state dictionary with all player/ball info
        </td>
    </tr>
</table>
"""

GRF_REWARD_HTML = """
<h4>Reward Structure</h4>
<table style="width:100%; border-collapse: collapse; margin: 10px 0;">
    <tr style="background-color: #f0f0f0;">
        <th style="border: 1px solid #ddd; padding: 6px;">Mode</th>
        <th style="border: 1px solid #ddd; padding: 6px;">Description</th>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>scoring</b> (default)</td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            +1 for scoring a goal, -1 for conceding. Sparse signal.
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>checkpoints</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            Dense reward: small positive signal when the ball crosses distance
            checkpoints toward the opponent's goal. Helps early training.
        </td>
    </tr>
    <tr>
        <td style="border: 1px solid #ddd; padding: 6px;"><b>scoring</b></td>
        <td style="border: 1px solid #ddd; padding: 6px;">
            Combined: both goal reward and checkpoint distance reward.
        </td>
    </tr>
</table>
"""
