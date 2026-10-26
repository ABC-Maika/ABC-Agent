"""Hot-seat human-vs-archetype game: observation decoder, rival-impression fold,
posture/confidence mapping, and the interactive layer (board, prompt, driver, reveal,
debrief). The rival is an NPC seat, so the human only ever sees its perceived (noised)
state; env imports are lazy so module load stays cheap.
"""

from retail_simulator.envs.hotseat.driver import (
    append_debrief,
    play_hotseat_game,
    print_rps_primer,
)

__all__ = [
    "append_debrief",
    "play_hotseat_game",
    "print_rps_primer",
]
