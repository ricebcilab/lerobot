"""The window a person (or a watcher of the expert) sees: the task, a HUD, and the
screens between blocks. Also paces the loop at the task rate and turns Esc or
closing the window into `session.Aborted`.
"""

import time

import numpy as np
import pygame

from .channel import is_active
from .session import Aborted

FIELD, HUD = 760, 124  # px: square task field, text band above it
BG, INK, INK_2 = (250, 250, 247), (31, 31, 30), (95, 94, 90)
TARGET, TARGET_EDGE, GOAL = (236, 235, 230), (201, 200, 193), (251, 227, 161)
CURSOR, PLAN, COMMAND = (42, 120, 214), (150, 190, 236), (235, 104, 52)
GOOD, BAD = (27, 150, 100), (210, 60, 60)
OBSTACLE, OBSTACLE_EDGE = (205, 204, 198), (150, 149, 144)
FREE_PLAY_KEYS = {pygame.K_1: ("mode", "policy"), pygame.K_2: ("mode", "teleop"), pygame.K_3: ("mode", "flow_control"),
                  pygame.K_4: ("mode", "frs"), pygame.K_LEFTBRACKET: ("depth", -1), pygame.K_RIGHTBRACKET: ("depth", +1),
                  pygame.K_MINUS: ("speed_gain", -0.1), pygame.K_EQUALS: ("speed_gain", +0.1),
                  pygame.K_KP_MINUS: ("speed_gain", -0.1), pygame.K_KP_PLUS: ("speed_gain", +0.1),
                  pygame.K_SPACE: ("skip", None)}


def to_px(p):
    return int(round(p[0] * FIELD)), int(round(HUD + (1 - p[1]) * FIELD))


class Skipped(Exception):
    """The trial was skipped (free play, space)."""


class Renderer:
    """blind=True hides the condition (for participants); free_play=True enables the
    setting keys and reports them through `requests`."""

    def __init__(self, title="Cursor shared autonomy", blind=True, free_play=False, ready_hold_s=0.3,
                 feedback_s=0.6, fps=50, auto_continue_s=None):
        self.blind, self.free_play = blind, free_play
        self.ready_hold_s, self.feedback_s, self.fps = ready_hold_s, feedback_s, fps
        self.auto_continue_s = auto_continue_s  # screens continue by themselves (watching the expert)
        self.requests = []       # free-play setting changes, (what, value)
        self.status_line = ""
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
                if self.free_play and event.key in FREE_PLAY_KEYS:
                    self.requests.append(FREE_PLAY_KEYS[event.key])
        if agent is not None and hasattr(agent, "keyboard"):
            held = pygame.key.get_pressed()
            agent.keyboard.update(held[pygame.K_LEFT], held[pygame.K_RIGHT], held[pygame.K_UP], held[pygame.K_DOWN],
                                  held[pygame.K_LSHIFT] or held[pygame.K_RSHIFT])
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

    def screen_message(self, title, lines, agent):
        """Show a screen until the person continues; redrawn so the device status stays current."""
        tic = time.monotonic()

        def draw():
            if hasattr(agent, "poll"):
                agent.poll()
            self.screen.fill(BG)
            self.text(title, (FIELD // 2, HUD + 200), big=True, center=True)
            for i, line in enumerate(lines):
                self.text(line, (FIELD // 2, HUD + 250 + 26 * i), center=True, color=INK_2)
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
        lines = [f"{info['n_practice']} practice trials, then {info['n_trials']} trials.",
                 "Push the stick to move the cursor onto the gold target and hold it there."]
        if info.get("resumed"):
            lines.append(f"Resuming after {info['resumed']} completed trials.")
        lines.append("Press a SpaceMouse button or Enter to start.")
        self.screen_message(f"Block {info['block']} of {info['n_blocks']}{arm}", lines, agent)

    def break_screen(self, info, agent):
        if info["last"]:
            self.screen_message("Done. Thank you!", ["Press a SpaceMouse button or Enter to finish."], agent)
        else:
            self.screen_message(f"End of block {info['block']} of {info['n_blocks']}",
                                ["Take a break if you like.", "Press a SpaceMouse button or Enter to continue."], agent)

    # ------------------------------------------------------------ trial phases
    def ready_phase(self, ctx):
        """Show the layout; start once the stick has been idle for ready_hold_s."""
        agent, idle_since = ctx["agent"], [None]

        def done(keys):
            idle = agent.idle
            idle_since[0] = None if not idle else (idle_since[0] if idle_since[0] is not None else time.monotonic())
            return idle_since[0] is not None and time.monotonic() - idle_since[0] >= self.ready_hold_s

        self.frames(agent, lambda: self.draw(ctx, banner="Get ready" if idle_since[0] is not None else "Release the stick",
                                             banner_color=INK_2), done)

    def frame(self, ctx):
        self.pump(ctx["agent"])
        if self.free_play and ("skip", None) in self.requests:
            self.requests.remove(("skip", None))
            raise Skipped()
        self.draw(ctx)
        self.clock.tick(self.fps)

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
        if self.blind:
            which = "practice" if hud.get("practice") else "trial"
            self.text(f"Block {hud.get('block', 1)} of {hud.get('n_blocks', 1)}   {which} {hud.get('trial', 1)} "
                      f"of {hud.get('n_trials', 1)}", (12, 8), big=True)
        else:
            self.text(hud.get("title", ""), (12, 8), big=True)
            self.text(hud.get("line2", ""), (12, 36), color=INK_2)
            self.text(hud.get("line3", ""), (12, 56), color=INK_2)
            self.text(hud.get("line4", ""), (12, 78))
        self.text(self.status_line, (12, 100), color=INK_2)
        if not done and is_active(ctx["served"][0]) and ctl.mode not in ("policy", "teleop"):
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
        plan = ctl.planned()[0]
        if len(plan) and not done:
            pygame.draw.lines(self.screen, PLAN, False, [to_px(env.pos[0])] + [to_px(p) for p in plan], 3)
        cursor = to_px(env.pos[0])
        if not done and is_active(ctx["served"][0]):
            pygame.draw.line(self.screen, COMMAND, cursor, to_px(env.pos[0] + 0.12 * ctx["served"][0]), 3)
        pygame.draw.circle(self.screen, CURSOR, cursor, 8)
        if banner:
            self.text(banner, (FIELD // 2, HUD + 24), big=True, color=banner_color, center=True)
        pygame.display.flip()

    def close(self):
        pygame.quit()
