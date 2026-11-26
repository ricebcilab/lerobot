"""The window a person (or a watcher of the expert) sees: the task, a HUD, and the
screens between blocks. Also paces the loop at the task rate and turns Esc or
closing the window into `session.Aborted`.
"""

import time

import numpy as np
import pygame

from lerobot.teleoperators.readout import is_active
from .session import Aborted

FIELD, HUD = 760, 124  # px: square task field, text band above it
BG, INK, INK_2 = (250, 250, 247), (31, 31, 30), (95, 94, 90)
TARGET, TARGET_EDGE, GOAL = (236, 235, 230), (201, 200, 193), (251, 227, 161)
CURSOR, PLAN, COMMAND = (42, 120, 214), (150, 190, 236), (235, 104, 52)
GUIDE = (112, 168, 150)  # a calibration reach's whole path, drawn ahead of the cursor in every window
GOOD, BAD = (27, 150, 100), (210, 60, 60)
OBSTACLE, OBSTACLE_EDGE = (205, 204, 198), (150, 149, 144)
MAX_BEHIND = 2  # frame periods a trial frame may run behind its deadline before the grid starts again (a stall)
# The keys a human agent's keyboard is told about, by the names its reader knows (`devices.KEY_PUSHES`, and shift).
HELD_KEYS = {"left": (pygame.K_LEFT,), "right": (pygame.K_RIGHT,), "up": (pygame.K_UP,), "down": (pygame.K_DOWN,),
             "shift": (pygame.K_LSHIFT, pygame.K_RSHIFT)}


def to_px(p):
    return int(round(p[0] * FIELD)), int(round(HUD + (1 - p[1]) * FIELD))


class Renderer:
    """blind=True hides the condition (for participants); unblinded, the arm's name
    shows beside the block and trial count. cues=False leaves out what tells the arms
    apart on screen: the served command's line, the planned path and the STEERING label
    (a blinded study shows the cursor, the targets and the obstacles)."""

    def __init__(self, title="Cursor shared autonomy", blind=True, ready_s=0.3,
                 feedback_s=0.6, fps=50, auto_continue_s=None, cues=True):
        self.blind, self.cues = blind, cues
        self.ready_s, self.feedback_s, self.fps = ready_s, feedback_s, fps
        self.auto_continue_s = auto_continue_s  # screens continue by themselves (watching the expert)
        self.deadline = None  # when the current trial frame is due to end (`frame`)
        pygame.init()
        self.screen = pygame.display.set_mode((FIELD, HUD + FIELD))
        pygame.display.set_caption(title)
        self.font = pygame.font.SysFont("dejavusans", 15)
        self.big = pygame.font.SysFont("dejavusans", 20, bold=True)
        self.clock = pygame.time.Clock()

    # ------------------------------------------------------------ events
    def pump(self, agent=None):
        """Handle window events; feed the keyboard to a human agent; return key presses."""
        pressed = []
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                raise Aborted()
            if event.type == pygame.KEYDOWN:
                if event.key in (pygame.K_ESCAPE, pygame.K_q):
                    raise Aborted()
                pressed.append(event.key)
        if agent is not None and hasattr(agent, "keyboard"):
            held = pygame.key.get_pressed()
            agent.keyboard.hold(name for name, keys in HELD_KEYS.items() if any(held[k] for k in keys))
        return pressed

    # ------------------------------------------------------------ screens
    def text(self, s, pos, big=False, color=INK, center=False):
        surf = (self.big if big else self.font).render(s, True, color)
        self.screen.blit(surf, surf.get_rect(center=pos) if center else pos)

    def frames(self, agent, draw, done):
        """Redraw at the frame rate until done(keys) is true; every frame handles the
        window events and the device (`pump`)."""
        while True:
            keys = self.pump(agent)
            if done(keys):
                return
            draw()
            self.clock.tick(self.fps)

    def page(self, title, lines):
        """A title and the lines under it, on an empty screen (not yet shown)."""
        self.screen.fill(BG)
        self.text(title, (FIELD // 2, HUD + 200), big=True, center=True)
        for i, line in enumerate(lines):
            self.text(line, (FIELD // 2, HUD + 250 + 26 * i), center=True, color=INK_2)

    def wait_screen(self, title, lines):
        """A screen that asks for nothing, shown once: a pause the session fills itself (a module being trained)."""
        self.page(title, lines)
        pygame.display.flip()

    def screen_message(self, title, lines, agent):
        """Show a screen until the person continues; redrawn so the device status stays current."""
        tic = time.monotonic()

        def draw():
            if hasattr(agent, "poll"):
                agent.poll()
            self.page(title, lines)
            if hasattr(agent, "status"):
                self.text(f"SpaceMouse: {agent.status}", (12, 100), color=INK_2)
            pygame.display.flip()

        def done(keys):
            return (any(k in (pygame.K_RETURN, pygame.K_KP_ENTER, pygame.K_SPACE) for k in keys)
                    or (hasattr(agent, "take_button") and agent.take_button())
                    or (self.auto_continue_s is not None and time.monotonic() - tic >= self.auto_continue_s))

        self.frames(agent, draw, done)

    def block_screen(self, info, agent):
        arm = "" if self.blind else f"  ({info['arm']})"
        if info.get("calibration"):  # the cursor plays the reaches by itself (`session.Calibration`)
            title = "Calibration"
            lines = [f"{info['n_practice']} practice reaches, then {info['n_trials']} reaches.",
                     "The cursor moves to the gold target by itself.",
                     "Push the stick toward the gold target while it moves, as if you were steering it."]
        else:
            title = f"Block {info['block']} of {info['n_blocks']}{arm}"
            lines = [f"{info['n_practice']} practice trials, then {info['n_trials']} trials.",
                     "Push the stick to move the cursor onto the gold target and hold it there."]
        if info.get("resumed"):
            lines.append(f"Resuming after {info['resumed']} completed trials.")
        lines.append("Press a SpaceMouse button or Enter to start.")
        self.screen_message(title, lines, agent)

    def section_screen(self, info, agent):
        """Before a task section of a visit over several tasks."""
        lines = ["This task has obstacles: steer around them. Touching one ends the trial." if info.get("obstacles")
                 else "This task has no obstacles.",
                 "Press a SpaceMouse button or Enter to continue."]
        self.screen_message(f"Task {info['position']} of {info['of']}", lines, agent)

    def break_screen(self, info, agent):
        if info["last"]:
            self.screen_message("Done. Thank you!", ["Press a SpaceMouse button or Enter to finish."], agent)
        elif info.get("section_end"):
            self.screen_message(f"End of block {info['block']} of {info['n_blocks']}",
                                ["That was the last block of this task. Take a longer break if you like.",
                                 "Press a SpaceMouse button or Enter to continue."], agent)
        else:
            self.screen_message(f"End of block {info['block']} of {info['n_blocks']}",
                                ["Take a break if you like.", "Press a SpaceMouse button or Enter to continue."], agent)

    # ------------------------------------------------------------ trial phases
    def ready_phase(self, ctx):
        """Show the layout under "Get ready" for ready_s, then start the trial, whatever the stick is doing."""
        tic = time.monotonic()
        self.frames(ctx["agent"], lambda: self.draw(ctx, banner="Get ready", banner_color=INK_2),
                    lambda keys: time.monotonic() - tic >= self.ready_s)
        self.begin_trial()

    def begin_trial(self):
        """The trial starts now: its frame k is due to end k frame periods from here (`frame`)."""
        self.deadline = time.monotonic()

    def frame(self, ctx):
        """One frame of a trial, paced by deadline: it waits until its own end on the trial's grid of frame periods,
        so a frame that ran late (a chunk computed in it) is followed by a shorter one and the trial keeps the task's
        rate on average, in wall-clock time as in steps. More than MAX_BEHIND periods late (a stall) starts the grid
        again from now, rather than a burst of short frames."""
        self.pump(ctx["agent"])
        self.draw(ctx)
        period, now = 1.0 / self.fps, time.monotonic()
        self.deadline = (now if self.deadline is None else self.deadline) + period
        if now - self.deadline > MAX_BEHIND * period:
            self.deadline = now
        elif now < self.deadline:
            time.sleep(self.deadline - now)

    def feedback_phase(self, ctx):
        env = ctx["env"]
        text = (f"Hit in {env.steps[0] * ctx['task'].dt:.2f} s" if env.success[0] else
                "Wrong target" if env.selected[0] >= 0 else "Collision" if env.collided[0] else "Timeout")
        tic = time.monotonic()
        self.frames(ctx["agent"], lambda: self.draw(ctx, banner=text, banner_color=GOOD if env.success[0] else BAD, done=True),
                    lambda keys: time.monotonic() - tic >= self.feedback_s)

    # ------------------------------------------------------------ drawing
    def draw(self, ctx, banner=None, banner_color=INK, done=False):
        env, ctl, layouts, task, hud = ctx["env"], ctx["ctl"], ctx["layouts"], ctx["task"], ctx["hud"]
        self.screen.fill(BG)
        which = "practice" if hud.get("practice") else "reach" if hud.get("calibration") else "trial"
        arm = "" if self.blind or not hud.get("arm") else f"   ({hud['arm']})"
        block = "Calibration" if hud.get("calibration") else f"Block {hud.get('block', 1)} of {hud.get('n_blocks', 1)}"
        self.text(f"{block}   {which} {hud.get('trial', 1)} of {hud.get('n_trials', 1)}{arm}", (12, 8), big=True)
        if self.cues and not done and is_active(ctx["served"][0]) and ctl.mode not in ("policy", "teleop", "playback"):
            self.text("STEERING", (FIELD - 100, 10), big=True, color=COMMAND)
        pygame.draw.line(self.screen, TARGET_EDGE, (0, HUD - 1), (FIELD, HUD - 1))

        for x, y, r in layouts.real_obstacles(0):
            pygame.draw.circle(self.screen, OBSTACLE, to_px((x, y)), int(r * FIELD))
            pygame.draw.circle(self.screen, BAD if done and env.collided[0] else OBSTACLE_EDGE, to_px((x, y)), int(r * FIELD), 2)
        radius = int(task.target_radius * FIELD)
        for j in np.flatnonzero(layouts.mask[0]):
            is_goal = j == layouts.goal[0]
            center = to_px(layouts.targets[0, j])
            pygame.draw.circle(self.screen, GOAL if is_goal else TARGET, center, radius)
            edge = INK if is_goal else TARGET_EDGE
            if done and j == env.selected[0]:
                edge = GOOD if env.success[0] else BAD
            pygame.draw.circle(self.screen, edge, center, radius, 3 if edge in (GOOD, BAD) else 2)
            if env.on[0] == j and not done:
                fill = int(radius * min(1.0, env.dwell[0] / task.dwell_steps))
                if fill > 0:
                    pygame.draw.circle(self.screen, PLAN, center, fill)
        guide = getattr(ctl, "guide", None)  # a calibration's: not an assist cue, so a blinded window draws it too
        if guide is not None and not done:
            pygame.draw.lines(self.screen, GUIDE, False, [to_px(p) for p in guide[0]], 3)
        plan = ctl.planned()[0] if self.cues else ()
        if len(plan) and not done:
            pygame.draw.lines(self.screen, PLAN, False, [to_px(env.pos[0])] + [to_px(p) for p in plan], 3)
        cursor = to_px(env.pos[0])
        if self.cues and not done and is_active(ctx["served"][0]):
            pygame.draw.line(self.screen, COMMAND, cursor, to_px(env.pos[0] + 0.12 * ctx["served"][0]), 3)
        pygame.draw.circle(self.screen, CURSOR, cursor, 8)
        if banner:
            self.text(banner, (FIELD // 2, HUD + 24), big=True, color=banner_color, center=True)
        pygame.display.flip()

    def close(self):
        pygame.quit()
