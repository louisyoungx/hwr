"""Top-down Pillow renderer for reference household simulation snapshots."""

from __future__ import annotations

import math
from dataclasses import dataclass

try:
    from PIL import Image, ImageDraw, ImageFont
except ModuleNotFoundError as error:  # pragma: no cover - depends on install extras
    raise ModuleNotFoundError(
        "video rendering requires Pillow; install the project with '.[video]'"
    ) from error

from hwr.render.rollout import RolloutFrame, RolloutTrace
from hwr.sim import Bounds, HouseholdTaskSpec, RobotSpec


TASK_NAMES = {
    "tidy_table/v1": "Tidy table",
    "sort_laundry/v1": "Sort laundry",
    "clear_dishes/v1": "Clear dishes",
}


@dataclass(frozen=True)
class RenderTheme:
    background: tuple[int, int, int] = (21, 25, 31)
    panel: tuple[int, int, int] = (30, 36, 44)
    floor: tuple[int, int, int] = (240, 237, 228)
    grid: tuple[int, int, int] = (216, 213, 205)
    border: tuple[int, int, int] = (102, 112, 124)
    text: tuple[int, int, int] = (241, 245, 249)
    muted: tuple[int, int, int] = (161, 171, 184)
    robot: tuple[int, int, int] = (45, 127, 249)
    arm: tuple[int, int, int] = (9, 68, 148)
    path: tuple[int, int, int] = (93, 173, 255)
    obstacle: tuple[int, int, int] = (113, 91, 77)
    zone: tuple[int, int, int] = (48, 170, 111)
    success: tuple[int, int, int] = (38, 190, 118)
    warning: tuple[int, int, int] = (245, 158, 11)


@dataclass(frozen=True)
class RoomTransform:
    bounds: Bounds
    left: float
    top: float
    scale: float
    room_height: float

    def point(self, x: float, y: float) -> tuple[int, int]:
        px = self.left + (x - self.bounds.min_x) * self.scale
        py = self.top + self.room_height - (y - self.bounds.min_y) * self.scale
        return round(px), round(py)

    def radius(self, value: float) -> int:
        return max(2, round(value * self.scale))


class RolloutRasterizer:
    """Render one or more synchronized rollout traces into RGB images."""

    def __init__(
        self,
        *,
        panel_width: int = 480,
        height: int = 720,
        theme: RenderTheme | None = None,
    ) -> None:
        if panel_width < 320 or height < 480:
            raise ValueError("render dimensions are too small")
        self.panel_width = panel_width
        self.height = height
        self.theme = theme or RenderTheme()
        self.title_font = _font(25)
        self.body_font = _font(17)
        self.small_font = _font(14)
        self.object_font = _font(13)

    def render_grid(
        self,
        traces: tuple[RolloutTrace, ...],
        tasks: tuple[HouseholdTaskSpec, ...],
        robot_spec: RobotSpec,
        frame_indices: tuple[int, ...],
    ) -> Image.Image:
        if not traces or not (len(traces) == len(tasks) == len(frame_indices)):
            raise ValueError("traces, tasks, and frame indices must have equal non-zero lengths")
        canvas = Image.new(
            "RGB",
            (self.panel_width * len(traces), self.height),
            self.theme.background,
        )
        for panel_index, (trace, task, frame_index) in enumerate(
            zip(traces, tasks, frame_indices, strict=True)
        ):
            panel = self.render_panel(trace, task, robot_spec, frame_index)
            canvas.paste(panel, (panel_index * self.panel_width, 0))
        return canvas

    def render_panel(
        self,
        trace: RolloutTrace,
        task: HouseholdTaskSpec,
        robot_spec: RobotSpec,
        frame_index: int,
    ) -> Image.Image:
        frame_index = min(max(frame_index, 0), len(trace.frames) - 1)
        frame = trace.frames[frame_index]
        image = Image.new("RGB", (self.panel_width, self.height), self.theme.panel)
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, self.panel_width - 1, self.height - 1), outline=self.theme.border)
        self._draw_header(draw, trace, task, frame)
        transform = self._room_transform(task.scene.bounds)
        self._draw_room(draw, task, transform)
        self._draw_path(draw, trace, frame_index, transform)
        self._draw_objects(draw, frame, transform)
        self._draw_robot(draw, frame, robot_spec, transform)
        self._draw_footer(draw, trace, frame)
        return image

    def _room_transform(self, bounds: Bounds) -> RoomTransform:
        area_left = 20
        area_top = 86
        area_width = self.panel_width - 40
        area_height = self.height - 218
        world_width = bounds.max_x - bounds.min_x
        world_height = bounds.max_y - bounds.min_y
        scale = min(area_width / world_width, area_height / world_height)
        room_width = world_width * scale
        room_height = world_height * scale
        left = area_left + (area_width - room_width) / 2
        top = area_top + (area_height - room_height) / 2
        return RoomTransform(bounds, left, top, scale, room_height)

    def _draw_header(
        self,
        draw: ImageDraw.ImageDraw,
        trace: RolloutTrace,
        task: HouseholdTaskSpec,
        frame: RolloutFrame,
    ) -> None:
        title = TASK_NAMES.get(task.task_id, task.task_id)
        draw.text((20, 16), title, font=self.title_font, fill=self.theme.text)
        elapsed = frame.snapshot.timestamp_ns / 1_000_000_000
        subtitle = f"learned policy  |  seed {trace.seed}  |  sim {elapsed:04.1f}s"
        draw.text((20, 52), subtitle, font=self.small_font, fill=self.theme.muted)

    def _draw_room(
        self,
        draw: ImageDraw.ImageDraw,
        task: HouseholdTaskSpec,
        transform: RoomTransform,
    ) -> None:
        bounds = task.scene.bounds
        top_left = transform.point(bounds.min_x, bounds.max_y)
        bottom_right = transform.point(bounds.max_x, bounds.min_y)
        draw.rectangle((*top_left, *bottom_right), fill=self.theme.floor, outline=self.theme.border, width=2)
        self._draw_grid(draw, transform)
        for obstacle in task.scene.obstacles:
            upper_left = transform.point(obstacle.min_x, obstacle.max_y)
            lower_right = transform.point(obstacle.max_x, obstacle.min_y)
            draw.rounded_rectangle(
                (*upper_left, *lower_right),
                radius=5,
                fill=self.theme.obstacle,
                outline=(75, 60, 52),
                width=2,
            )
            draw.text(
                (upper_left[0] + 5, upper_left[1] + 5),
                obstacle.obstacle_id,
                font=self.object_font,
                fill=(250, 245, 240),
            )
        for zone in task.scene.zones:
            center = transform.point(zone.center_x, zone.center_y)
            radius = transform.radius(zone.radius)
            box = _circle_box(center, radius)
            draw.ellipse(box, fill=(205, 235, 218), outline=self.theme.zone, width=4)
            label_position = (center[0] - radius, center[1] + radius + 4)
            draw.text(label_position, zone.zone_id, font=self.object_font, fill=(27, 89, 60))

    def _draw_grid(self, draw: ImageDraw.ImageDraw, transform: RoomTransform) -> None:
        bounds = transform.bounds
        first_x = math.ceil(bounds.min_x * 2) / 2
        first_y = math.ceil(bounds.min_y * 2) / 2
        x = first_x
        while x <= bounds.max_x:
            start = transform.point(x, bounds.min_y)
            end = transform.point(x, bounds.max_y)
            draw.line((*start, *end), fill=self.theme.grid, width=1)
            x += 0.5
        y = first_y
        while y <= bounds.max_y:
            start = transform.point(bounds.min_x, y)
            end = transform.point(bounds.max_x, y)
            draw.line((*start, *end), fill=self.theme.grid, width=1)
            y += 0.5

    def _draw_path(
        self,
        draw: ImageDraw.ImageDraw,
        trace: RolloutTrace,
        frame_index: int,
        transform: RoomTransform,
    ) -> None:
        points = [
            transform.point(item.snapshot.robot.x, item.snapshot.robot.y)
            for item in trace.frames[: frame_index + 1 : 2]
        ]
        current = trace.frames[frame_index].snapshot.robot
        current_point = transform.point(current.x, current.y)
        if not points or points[-1] != current_point:
            points.append(current_point)
        if len(points) >= 2:
            draw.line(points, fill=self.theme.path, width=3, joint="curve")

    def _draw_objects(
        self,
        draw: ImageDraw.ImageDraw,
        frame: RolloutFrame,
        transform: RoomTransform,
    ) -> None:
        colors = ((234, 88, 75), (155, 103, 224), (239, 177, 61), (55, 156, 173))
        for index, item in enumerate(frame.snapshot.objects):
            center = transform.point(item.x, item.y)
            radius = max(transform.radius(item.radius), 7)
            fill = (124, 199, 156) if item.placed else colors[index % len(colors)]
            draw.ellipse(_circle_box(center, radius), fill=fill, outline=(56, 51, 47), width=2)
            label = item.object_id.replace("_", " ")
            draw.text(
                (center[0] + radius + 4, center[1] - 8),
                label,
                font=self.object_font,
                fill=(40, 44, 48),
            )
            if item.placed:
                draw.line(
                    (center[0] - 4, center[1], center[0] - 1, center[1] + 4, center[0] + 6, center[1] - 5),
                    fill=(18, 91, 57),
                    width=2,
                )

    def _draw_robot(
        self,
        draw: ImageDraw.ImageDraw,
        frame: RolloutFrame,
        robot_spec: RobotSpec,
        transform: RoomTransform,
    ) -> None:
        robot = frame.snapshot.robot
        center = transform.point(robot.x, robot.y)
        radius = transform.radius(robot_spec.base_radius)
        draw.ellipse(_circle_box(center, radius), fill=self.theme.robot, outline=(12, 55, 120), width=3)
        nose = transform.point(
            robot.x + robot_spec.base_radius * 1.15 * math.cos(robot.heading),
            robot.y + robot_spec.base_radius * 1.15 * math.sin(robot.heading),
        )
        draw.line((*center, *nose), fill=(248, 250, 252), width=4)
        endpoint = transform.point(robot.end_effector_x, robot.end_effector_y)
        draw.line((*center, *endpoint), fill=self.theme.arm, width=6)
        gripper_radius = 7 if robot.gripper >= 0.5 else 10
        draw.ellipse(_circle_box(endpoint, gripper_radius), outline=self.theme.arm, width=3)

    def _draw_footer(
        self,
        draw: ImageDraw.ImageDraw,
        trace: RolloutTrace,
        frame: RolloutFrame,
    ) -> None:
        snapshot = frame.snapshot
        placed = sum(item.placed for item in snapshot.objects)
        carrying = snapshot.robot.carrying_object_id or "none"
        draw.text(
            (20, self.height - 112),
            f"stage {snapshot.task_stage}   carrying {carrying}",
            font=self.body_font,
            fill=self.theme.text,
        )
        draw.text(
            (20, self.height - 80),
            f"step {snapshot.steps:03d}   placed {placed}/{len(snapshot.objects)}   collisions {snapshot.collisions}",
            font=self.small_font,
            fill=self.theme.muted,
        )
        status, color = _frame_status(trace, frame)
        draw.rounded_rectangle(
            (20, self.height - 46, self.panel_width - 20, self.height - 14),
            radius=8,
            fill=color,
        )
        draw.text((32, self.height - 41), status, font=self.body_font, fill=(255, 255, 255))


def _frame_status(trace: RolloutTrace, frame: RolloutFrame) -> tuple[str, tuple[int, int, int]]:
    event_names = {event.event_type for event in frame.events}
    if "task_succeeded" in event_names:
        return "SUCCESS - task completed", (38, 154, 98)
    if "task_failed" in event_names:
        return f"FAILED - {trace.result.reason}", (190, 55, 55)
    for event in reversed(frame.events):
        if event.event_type == "object_grasped":
            return f"grasped {event.details['object_id']}", (196, 123, 28)
        if event.event_type == "object_released":
            return f"released {event.details['object_id']}", (46, 119, 181)
    return "closed-loop inference running", (54, 92, 132)


def _circle_box(center: tuple[int, int], radius: int) -> tuple[int, int, int, int]:
    return (
        center[0] - radius,
        center[1] - radius,
        center[0] + radius,
        center[1] + radius,
    )


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:  # pragma: no cover - platform font availability
        return ImageFont.load_default(size=size)
