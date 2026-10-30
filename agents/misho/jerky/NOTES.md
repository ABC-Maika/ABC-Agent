# jerky

- **Owner:** misho. **Algorithm:** TQC (`MlpPolicy`, 512x512, sb3-contrib), untrained after cloning: the weights are a behaviour clone of a scripted recipe. Research lever capped at 0.4 (Box bound in the saved action space). Stock checkpoint (`FloatSchedule`, no function saved by value), so battle rooms accept it.
- **Checkpoint on main: "lab-opt".** The clone of the lever lab's best recipe (`agents/misho/jacky-t/lever_lab.py`): price 2.0, marketing, assortment and promotion 1.0, service 0.2, research 0.1, loyalty 0.4 (inert), wage 0.45 + 4 x (1 - happiness), warehouse 0, METRO then AFFLUENT regions highest number first, up to 10 stores.
- **Recipe:** `lab_collect.py play opt` (30 games with the lab seat in Jerky's seat) and `lab_collect.py dagger opt` (Jerky's states relabelled by the lab), then `mimic.py bc ... --new-net 512,512 --max-rank 2 --opening-boost 8 --expand-boost 8 --expansion-ce 0.5 --research-cap 0.4 --log-std -2 --ent-init 0.02 --epochs 30`. No RL steps on top: every 10k-step RL session started from this clone (sessions 13, 14, 15) scored lower.

## Where it stands
- **Organizer site battle (6 rotations, Krischo, Jack, Orcho, Orcho-1, YafiPafi):** 1st, 595,937 (Krischo 590,266). Five of six matches positive (+111k to +219k), one at -169,912 (home Stara Zagora).
- **Local battle format, seed 7001** (Orcho v5, Krischo MLP v10-rl, KMlp8, Yavkata SAC-v2): +122,626 mean, 5 of 6 wins. Against the newest agents (Orcho v7 and v6.2 in one field) it is negative (-32k mean): it plays Orcho's own opening, so it is a twin of any Orcho agent.

## Known weaknesses
- **Bimodal.** It opens its 4th store at round 3 with about 65 cash left; if rivals share its first regions it misses the 5,000 gate, stays at 3 to 4 stores, and the world's restructuring (cash below -10,000 for 50 rounds) leaves it at 3 stores losing about -250 per round (-180k to -250k a match). The first rounds look the same in winning and losing games; the split shows after round 150.
- **Twin of Orcho.** Two agents with the same opening split every region. A lowest-region-first opening (tested in the scripts) has fewer losing games (1 of 18 against 2) and more wins (10 against 7) in one-agent-per-team fields, at a lower mean (+260k against +339k).
- **No memory, no rival identity.** It cannot see who leads or any rival's profit.
- Tried and rejected: cash buffers, a rival-aware opening switch (no usable early signal), adaptive recipe switching, late-game price cuts (about +2k), spoiler tactics (profit moves to the other rivals).
