"""Replay recorded AROOD training episodes without creating a simulator or loading a policy.

A.py appends actual pre/post vehicle states and decision context to
figures/episode_trajectories.jsonl. This script only reads that archive and
animates the recorded positions; it never calls env.reset(), env.step(), or a policy.
"""
import argparse
import json
import math
import os
from collections import OrderedDict
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.patches import Polygon
import C_Video


DEFAULT_ARCHIVE = C_Video.EPISODE_TRAJECTORY_PATH
gif_save_path = 'figure/videos/replay_13.gif'
Episode = 94


def load_archive(path):
    records = []
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Trajectory archive not found: {os.path.abspath(path)}\n"
            "Run the instrumented A.py first. Episodes that were not recorded "
            "cannot be reconstructed retrospectively."
        )
    with open(path, "r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[ARCHIVE WARNING] skipping malformed line {line_number}: {exc}")
    return records


def grouped_episodes(records):
    episodes = OrderedDict()
    for record in records:
        key = record.get("episode_key")
        if not key:
            continue
        group = episodes.setdefault(key, {"start": None, "transitions": []})
        if record.get("record_type") == "episode_start":
            group["start"] = record
        elif record.get("record_type") == "transition":
            group["transitions"].append(record)
    return episodes


def list_episodes(episodes):
    print("Recorded episodes:")
    for key, episode in episodes.items():
        start = episode["start"] or {}
        transitions = sorted(episode["transitions"], key=lambda x: int(x.get("step", 0)))
        collision = any(
            bool((frame.get("step_info") or {}).get("collision", False))
            for frame in transitions
        )
        print(
            f"  phase={start.get('phase', '?'):<20} "
            f"episode={start.get('episode_number', '?')!s:<5} "
            f"steps={len(transitions):<4} collision={collision!s:<5} "
            f"run_id={start.get('run_id', '?')}  key={key}"
        )


def find_episode(episodes, phase, episode_number, run_id=None):
    matches = []
    for key, data in episodes.items():
        start = data["start"]
        if not start:
            continue
        if str(start.get("phase")) != str(phase):
            continue
        if int(start.get("episode_number", -1)) != int(episode_number):
            continue
        if run_id and start.get("run_id") != run_id:
            continue
        matches.append((key, data))
    if not matches:
        available = [
            f"{(v['start'] or {}).get('phase')}:{(v['start'] or {}).get('episode_number')} "
            f"(run={(v['start'] or {}).get('run_id')})"
            for v in episodes.values() if v["start"]
        ]
        raise LookupError(
            f"No recorded episode matched phase={phase!r}, episode={episode_number}, "
            f"run_id={run_id!r}. Available starts: {available[-20:]}"
        )
    # JSONL is append-only; when run_id is omitted, select the most recently
    # recorded matching episode rather than accidentally replaying an older run.
    return matches[-1]


def build_frames(episode):
    start = episode["start"]
    if not start:
        raise ValueError("Episode start record is missing; this episode cannot be replayed.")
    frames = [{
        "step": 0,
        "vehicles": start.get("vehicles", []),
        "action": "INITIAL STATE",
        "target_lane": None,
        "reward": None,
        "safety_info": {},
        "step_info": {},
        "state": start.get("initial_state"),
    }]
    for record in sorted(episode["transitions"], key=lambda x: int(x.get("step", 0))):
        step_info = record.get("step_info") or {}
        action = record.get("executed_high_level_action", [])
        lane_action = "CHANGE" if len(action) > 1 and int(round(float(action[1]))) == 1 else "KEEP"
        frames.append({
            "step": int(record.get("step", len(frames))),
            "vehicles": record.get("vehicles_after", []),
            "action": lane_action,
            "target_lane": record.get("selected_target_lane"),
            "reward": record.get("reward"),
            "safety_info": record.get("safety_info") or {},
            "step_info": step_info,
            "state": record.get("state_after"),
        })
    if len(frames) == 1:
        raise ValueError("This episode has no recorded transitions.")
    return frames


def rotated_rectangle(position, heading, length, width):
    """Return four corners for a vehicle rectangle in world coordinates."""
    x, y = float(position[0]), float(position[1])
    c, s = math.cos(float(heading)), math.sin(float(heading))
    local = [
        (-length / 2, -width / 2),
        (length / 2, -width / 2),
        (length / 2, width / 2),
        (-length / 2, width / 2),
    ]
    return [
        (x + px * c - py * s, y + px * s + py * c)
        for px, py in local
    ]


def replay_episode(start, frames, playback_speed=1.0, save_path=gif_save_path):
    fig, ax = plt.subplots(figsize=(13, 6))
    fig.canvas.manager.set_window_title(
        f"AROOD recorded episode — {start.get('phase')} #{start.get('episode_number')}"
    )
    lane_count = int(start.get("num_lanes", 3))
    lane_width = 4.0
    lane_centres = [i * lane_width for i in range(lane_count)]
    lane_edges = [-lane_width / 2 + i * lane_width for i in range(lane_count + 1)]

    for edge in lane_edges:
        ax.axhline(edge, linewidth=1.0, linestyle="-" if edge in (lane_edges[0], lane_edges[-1]) else "--")
    ax.set_xlabel("Road longitudinal position x (m)")
    ax.set_ylabel("Lateral position y (m)")
    ax.set_ylim(lane_edges[0] - 1.5, lane_edges[-1] + 1.5)
    ax.set_yticks(lane_centres, [f"Lane {i}" for i in range(lane_count)])
    ax.grid(axis="x", alpha=0.2)
    ax.set_title("Recorded trajectory replay — no policy inference or simulator stepping")

    vehicle_patches = []
    trail_x, trail_y = [], []
    trail, = ax.plot([], [], linewidth=1.5, linestyle=":")
    info_text = ax.text(
        0.015, 0.985, "", transform=ax.transAxes, va="top", ha="left",
        fontsize=9, bbox={"boxstyle": "round", "alpha": 0.85}
    )
    status_text = ax.text(
        0.985, 0.985, "", transform=ax.transAxes, va="top", ha="right",
        fontsize=10, weight="bold"
    )
    ego_label = ax.text(
        0.0, 0.0, "EGO", ha="center", va="bottom",
        fontsize=9, weight="bold", color="#0755a3", zorder=10
    )

    def draw(frame_index):
        frame = frames[frame_index]
        vehicles = frame.get("vehicles", [])
        ego = next((v for v in vehicles if v.get("is_ego")), None)
        if ego is None and vehicles:
            ego = vehicles[0]
        if ego is not None:
            ego_pos = ego.get("position_xy_m", [0.0, 0.0])
            centre_x = float(ego_pos[0])
            trail_x.append(centre_x)
            trail_y.append(float(ego_pos[1]))
            ax.set_xlim(centre_x - 90.0, centre_x + 130.0)
            trail.set_data(trail_x, trail_y)

        while vehicle_patches:
            patch = vehicle_patches.pop()
            patch.remove()

        for vehicle in vehicles:
            pos = vehicle.get("position_xy_m")
            if not pos or len(pos) < 2:
                continue
            is_ego = bool(vehicle.get("is_ego", False))
            try:
                heading = float(vehicle.get("heading_rad", 0.0))
                length = float(vehicle.get("length_m", 5.0))
                width = float(vehicle.get("width_m", 2.0))
                corners = rotated_rectangle(pos, heading, length, width)
            except (TypeError, ValueError):
                continue
            patch = Polygon(
                corners, closed=True,
                facecolor="#1687ff" if is_ego else "#8c969f",
                edgecolor="#063d78" if is_ego else "#353a40",
                linewidth=1.4 if is_ego else 0.8,
                alpha=0.95 if is_ego else 0.78,
                zorder=5 if is_ego else 3,
            )
            ax.add_patch(patch)
            vehicle_patches.append(patch)
            if is_ego:
                ego_label.set_position((float(pos[0]), float(pos[1]) + width / 2 + 0.3))
                ego_label.set_visible(True)

        if ego is None:
            ego_label.set_visible(False)

        safety = frame.get("safety_info") or {}
        step_info = frame.get("step_info") or {}
        reward = frame.get("reward")
        reward_text = "n/a" if reward is None else f"{float(reward):+.3f}"
        info_text.set_text(
            f"phase: {start.get('phase')}   episode: {start.get('episode_number')}\n"
            f"step: {frame.get('step')} / {len(frames) - 1}   "
            f"action: {frame.get('action')}   target lane: {frame.get('target_lane')}\n"
            f"reward: {reward_text}   "
            f"lane change: {step_info.get('lane_change_completed_instantly', False)}\n"
            f"blocked: {safety.get('lane_change_blocked', False)}   "
            f"reason: {safety.get('lane_change_rejection_reason', 'n/a')}"
        )
        if bool(step_info.get("collision", False)) or any(
            bool(v.get("crashed", False)) for v in vehicles if v.get("is_ego")
        ):
            status_text.set_text("COLLISION")
            status_text.set_color("red")
        else:
            status_text.set_text("RECORDED REPLAY")
            status_text.set_color("#087443")
        return [trail, info_text, status_text, ego_label, *vehicle_patches]

    interval = max(1, int(1000 / max(0.1, float(start.get("policy_frequency_hz", 5)) * playback_speed)))
    animation = FuncAnimation(
        fig, draw, frames=len(frames), interval=interval,
        blit=False, repeat=False, cache_frame_data=False
    )
    fig.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
        writer = "pillow"
        animation.save(save_path, writer=writer, dpi=120)
        print(f"[REPLAY] animation saved: {os.path.abspath(save_path)}")
    # Keep a reference alive until show() completes.
    plt.show()
    return animation


def main():
    parser = argparse.ArgumentParser(
        description="Replay a previously recorded AROOD episode; no new environment is created."
    )
    parser.add_argument("--archive", default=DEFAULT_ARCHIVE, help=f"JSONL archive (default: {DEFAULT_ARCHIVE}).")
    parser.add_argument("--phase", default="pure_rl", choices=("pure_rl", "reference_collection"),
                        help="Training phase to replay (default: pure_rl).")
    parser.add_argument("--episode", type=int, default=Episode, help="1-based episode number within the selected phase.")
    parser.add_argument("--run-id", default=None, help="Optional exact run_id; without it, the newest matching episode is used.")
    parser.add_argument("--speed", type=float, default=1.0, help="Playback speed multiplier (default: 1.0).")
    parser.add_argument("--list", action="store_true", help="List recorded episodes and exit.")
    parser.add_argument("--save", default=None, help="Optional output .mp4 or .gif path.")
    args = parser.parse_args()

    records = load_archive(args.archive)
    episodes = grouped_episodes(records)
    if args.list:
        list_episodes(episodes)
        return
    key, episode = find_episode(episodes, args.phase, args.episode, args.run_id)
    start = episode["start"]
    frames = build_frames(episode)
    print(f"[REPLAY] archive: {os.path.abspath(args.archive)}")
    print(f"[REPLAY] episode key: {key}")
    print(f"[REPLAY] recorded frames: {len(frames)} (including initial state)")
    print("[REPLAY] policy inference: disabled; simulator stepping: disabled")
    replay_episode(start, frames, playback_speed=args.speed, save_path=gif_save_path)


if __name__ == "__main__":
    main()


'''
How to use it:
conda activate highway_rl
cd /d F:\Doctoral_2

# List recorded episodes
python visible_simulation.py --list

# Replay online RL episode 1
python visible_simulation.py --phase pure_rl --episode 1

# Replay reference-collection episode 12
python visible_simulation.py --phase reference_collection --episode 12

# Save a replay as a GIF
python visible_simulation.py --phase pure_rl --episode 1 --save save_path

'''