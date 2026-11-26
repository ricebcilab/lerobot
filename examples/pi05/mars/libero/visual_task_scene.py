"""Disk and solid utensil-holder variants of the LIBERO bowl pilot."""

import xml.etree.ElementTree as ET

import numpy as np
from torch.utils._pytree import tree_map

RADIUS = 0.028  # metres, scored at the bowl centre
BOWL_RADIUS = 0.050  # approximate outer rim; used only to label straight-line layouts
BOWL = "akita_black_bowl_1"
DESTINATIONS = {1: "flat_stove_1", 8: "plate_1"}


def segment_distance(start, end, point):
    delta = end - start
    t = np.clip(np.dot(point - start, delta) / max(np.dot(delta, delta), 1e-12), 0, 1)
    return np.linalg.norm(start + t * delta - point)


def marker_xml(xml):
    root = ET.fromstring(xml)
    body = ET.SubElement(root.find("worldbody"), "body", name="mars_marker", mocap="true")
    ET.SubElement(
        body,
        "geom",
        type="cylinder",
        size=f"{RADIUS} 0.001",
        group="1",
        rgba="1 0.05 0.55 1",
        contype="0",
        conaffinity="0",
    )
    return ET.tostring(root, encoding="unicode")


def holder_xml(xml):
    """A fixed ceramic crock with a dark inset and two wooden utensils. Dimensions are metres."""
    root = ET.fromstring(xml)
    body = ET.SubElement(root.find("worldbody"), "body", name="mars_marker", mocap="true", pos="0 0 -1")
    ceramic = "0.26 0.54 0.55 1"
    wood = "0.73 0.45 0.22 1"
    parts = [
        ("cup", "cylinder", "0 0 .057", ".028 .057", ceramic),
        ("rim", "cylinder", "0 0 .114", ".029 .002", ceramic),
        ("inside", "cylinder", "0 0 .116", ".024 .0005", ".12 .18 .18 1"),
        ("spoon_handle", "capsule", ".010 0 .136", ".0035 .025", wood),
        ("spoon", "ellipsoid", ".010 0 .165", ".011 .004 .016", wood),
        ("turner_handle", "capsule", "-.012 .003 .14", ".003 .028", wood),
        ("turner", "box", "-.012 .003 .172", ".009 .0025 .012", wood),
    ]
    for name, kind, pos, size, color in parts:
        ET.SubElement(
            body,
            "geom",
            name="holder_" + name,
            type=kind,
            pos=pos,
            size=size,
            rgba=color,
            group="1",
            contype="1",
            conaffinity="1",
        )
    return ET.tostring(root, encoding="unicode")


def two_obstacle_xml(xml):
    root = ET.fromstring(holder_xml(xml))
    mill = ET.SubElement(root.find("worldbody"), "body", name="mars_pepper_mill", mocap="true", pos="0 0 -2")
    for label, kind, z, size in (
        ("base", "cylinder", ".014", ".028 .014"),
        ("waist", "capsule", ".060", ".018 .035"),
        ("top", "ellipsoid", ".119", ".026 .026 .031"),
        ("knob", "sphere", ".152", ".006"),
    ):
        ET.SubElement(
            mill,
            "geom",
            name="mill_" + label,
            type=kind,
            pos="0 0 " + z,
            size=size,
            rgba=".31 .13 .055 1" if label != "knob" else ".7 .7 .7 1",
            group="1",
            contype="1",
            conaffinity="1",
        )
    return ET.tostring(root, encoding="unicode")


class PilotScene:
    def __init__(self, env, obstacle="disk"):
        self.obstacle = obstacle
        self.env = env.unwrapped
        self.env._ensure_env()
        self.domain = self.env._env.env
        self.domain.set_xml_processor(
            {"disk": marker_xml, "holder": holder_xml, "two_obstacles": two_obstacle_xml}[obstacle]
        )
        self.marker_body = "mars_marker"

    def position(self, name):
        model = self.domain.sim.model
        obj = self.domain.objects_dict.get(name) or self.domain.fixtures_dict[name]
        return self.domain.sim.data.body_xpos[model.body_name2id(obj.root_body)].copy()

    def reset(self, spec, layout_seed, route):
        self.env.reset(seed=spec.env_seed, options={"init_state_id": spec.init_state_id})
        self.start = self.position(BOWL)
        self.goal = self.position(DESTINATIONS[spec.task_id])
        self.first_contact = None
        self.control_step = 0
        if self.obstacle in ("holder", "two_obstacles"):
            sim = self.domain.sim
            names = [sim.model.geom_id2name(i) or "" for i in range(sim.model.ngeom)]
            bodies = {"mars_marker": "utensil_holder"}
            if self.obstacle == "two_obstacles":
                bodies["mars_pepper_mill"] = "pepper_mill"
            body_labels = {sim.model.body_name2id(k): v for k, v in bodies.items()}
            self.obstacle_geoms = {
                i: body_labels[b] for i, b in enumerate(sim.model.geom_bodyid) if b in body_labels
            }
            self.contact_groups = {
                i: "bowl" if name.startswith(BOWL) else "robot"
                for i, name in enumerate(names)
                if name.startswith((BOWL, "robot0_", "gripper0_"))
            }
        if self.obstacle == "two_obstacles":
            # 'route' is the holder assignment, independent of the requested goal.
            self.centers = {}
            for body in ("mars_marker", "mars_pepper_mill"):
                idx = sim.model.body_mocapid[sim.model.body_name2id(body)]
                sim.data.mocap_pos[idx] = [0, 0, -2]
            sim.forward()
            for destination in DESTINATIONS:
                self.marker_body = "mars_marker" if destination == route else "mars_pepper_mill"
                self.centers[destination] = self.layout_center(
                    destination, np.random.SeedSequence([layout_seed, destination])
                )
            self.marker_body = "mars_marker" if spec.task_id == route else "mars_pepper_mill"
            self.center = self.centers[spec.task_id]
        else:
            if self.obstacle == "holder":
                self.alternate = self.layout_center(8 if route == 1 else 1, layout_seed)
            self.center = self.layout_center(route, layout_seed)
        self.waypoint = None
        self.detoured = False
        self.violation = False
        self.previous = self.start[:2]
        obs = self.render_at(self.center)
        if self.obstacle in ("holder", "two_obstacles"):
            # Robosuite integrates several physics steps per control step. Latch brief contacts too.
            sim = self.domain.sim
            step = getattr(sim, "_pilot_original_step", sim.step)
            sim._pilot_original_step = step

            def monitored_step(*args, **kwargs):
                step(*args, **kwargs)
                if self.first_contact is None:
                    for obstacle_id, gid in self.obstacle_contacts():
                        if gid in self.contact_groups:
                            self.first_contact = {
                                "step": self.control_step,
                                "with": self.contact_groups[gid],
                                "geom": names[gid],
                                "obstacle": self.obstacle_geoms[obstacle_id],
                                "link": sim.model.body_id2name(sim.model.geom_bodyid[gid]),
                            }
                            self.violation = True
                            break

            sim.step = monitored_step
        return obs

    def obstacle_contacts(self):
        data = self.domain.sim.data
        for contact in data.contact[: data.ncon]:
            if contact.dist <= 0:
                if contact.geom1 in self.obstacle_geoms:
                    yield contact.geom1, contact.geom2
                elif contact.geom2 in self.obstacle_geoms:
                    yield contact.geom2, contact.geom1

    def layout_center(self, route, layout_seed):
        # Each destination sees both placement distributions in equal numbers.
        anchor = self.position(DESTINATIONS[route])
        rng = np.random.default_rng(layout_seed)
        delta = anchor[:2] - self.start[:2]
        side = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
        holder = self.obstacle in ("holder", "two_obstacles")
        width = RADIUS + BOWL_RADIUS - 0.003 if holder else 0.012
        for _ in range(1000 if holder else 100):
            center = self.start[:2] + rng.uniform(0.38, 0.62) * delta + rng.uniform(-width, width) * side
            if all(
                np.linalg.norm(center - p) > RADIUS + 0.012
                for p in (self.start[:2], *(self.position(n)[:2] for n in DESTINATIONS.values()))
            ):
                if holder:
                    other = self.position(DESTINATIONS[8 if route == 1 else 1])
                    if segment_distance(self.start[:2], other[:2], center) <= RADIUS + BOWL_RADIUS:
                        continue
                    if self.obstacle == "two_obstacles" and any(
                        np.linalg.norm(center - placed) < 2 * RADIUS + 0.012
                        for placed in self.centers.values()
                    ):
                        continue
                    self.move_marker(center)
                    # The holder stands just above the table; other reset contacts invalidate the layout.
                    if next(self.obstacle_contacts(), None) is not None:
                        continue
                break
        else:
            raise ValueError(
                f"No feasible {self.obstacle} placement for route {route} at {self.start} toward {anchor}"
            )
        return center

    def move_marker(self, center):
        sim = self.domain.sim
        idx = sim.model.body_mocapid[sim.model.body_name2id(self.marker_body)]
        sim.data.mocap_pos[idx] = [
            *center,
            self.domain.table_offset[2] + (0.0001 if self.obstacle in ("holder", "two_obstacles") else 0.002),
        ]
        sim.forward()

    def render_at(self, center):
        self.move_marker(center)
        return self.observation(self.domain._get_observations(force_update=True))

    def observation(self, raw):
        return tree_map(lambda x: np.expand_dims(x, 0), self.env._format_raw_obs(raw))

    @property
    def blocked(self):
        return segment_distance(self.start[:2], self.goal[:2], self.center) <= (
            RADIUS + BOWL_RADIUS if self.obstacle in ("holder", "two_obstacles") else RADIUS
        )

    def step(self, action):
        # Sample the actual terminal bowl pose before the Gym wrapper's success auto-reset.
        raw, _, done, _ = self.env._env.step(action)
        bowl = self.position(BOWL)[:2]
        if (
            self.obstacle in ("holder", "two_obstacles")
            and self.first_contact
            and "grasped" not in self.first_contact
        ):
            self.first_contact.update(
                grasped=bool(
                    self.domain._check_grasp(
                        self.domain.robots[0].gripper, self.domain.objects_dict[BOWL].contact_geoms
                    )
                ),
                finger_positions=raw["robot0_gripper_qpos"].ravel().tolist(),
            )
        self.control_step += 1
        if self.obstacle == "disk":
            self.violation |= bool(segment_distance(self.previous, bowl, self.center) <= RADIUS)
        self.previous = bowl
        success = self.env._env.check_success()
        return self.observation(raw), bool(done or success), bool(success)

    def start_pick_trace(self):
        self.initial_z = {name: self.position(name)[2] for name in self.domain.objects_dict}
        self.picked = []
        self.max_lift = dict.fromkeys(self.initial_z, 0.0)

    def track_picks(self):
        for name, initial in self.initial_z.items():
            lift = float(self.position(name)[2] - initial)
            self.max_lift[name] = max(lift, self.max_lift[name])
            if (
                lift >= 0.025
                and name not in self.picked
                and self.domain._check_grasp(
                    self.domain.robots[0].gripper, self.domain.objects_dict[name].contact_geoms
                )
            ):
                self.picked.append(name)

    def pick_result(self):
        states = self.domain.object_states_dict
        targets = {"plate": "plate_1", "stove": "flat_stove_1_cook_region"}
        return {
            "picked": self.picked,
            "max_lift_m": self.max_lift,
            "bowl_final_on": [
                name for name, target in targets.items() if states[target].check_ontop(states[BOWL])
            ],
        }

    def teacher_action(self, action):
        """Keep the grasp while traversing a clear side waypoint, then resume task-prompted placement."""
        bowl = self.position(BOWL)
        if self.waypoint is None:
            if self.detoured or bowl[2] < self.start[2] + 0.025:
                return action
            if segment_distance(bowl[:2], self.goal[:2], self.center) > RADIUS + (
                0.045 if self.obstacle in ("holder", "two_obstacles") else 0.01
            ):
                return action
            delta = self.goal[:2] - bowl[:2]
            side = np.array([-delta[1], delta[0]]) / max(np.linalg.norm(delta), 1e-8)
            candidates = [
                self.center
                + sign * side * (RADIUS + (0.092 if self.obstacle in ("holder", "two_obstacles") else 0.055))
                for sign in (-1, 1)
            ]
            self.waypoint = min(
                candidates, key=lambda p: np.linalg.norm(p - bowl[:2]) + np.linalg.norm(self.goal[:2] - p)
            )
        if np.linalg.norm(self.waypoint - bowl[:2]) < 0.015:
            self.detoured = True
            self.waypoint = None
            return action
        waypoint = np.r_[self.waypoint, max(self.goal[2] + 0.10, self.start[2] + 0.10)]
        action = action.copy()
        action[:3] = np.clip((waypoint - bowl) * 12, -0.65, 0.65)
        action[3:6] = 0
        action[6] = 1  # LIBERO positive gripper closes
        return action
