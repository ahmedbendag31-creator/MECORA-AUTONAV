
import math
import random
import joblib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle


# ============================================================
# MECORA V5 - Autonomous Navigation Simulator
# ============================================================

WIDTH = 10.0
HEIGHT = 7.0

ROBOT_RADIUS = 0.25
TARGET_RADIUS = 0.35

MAX_SENSOR_DISTANCE = 5.0

# Movement
FORWARD_SPEED = 0.075
TURN_SPEED = math.radians(12)

# Safety
DANGER_FRONT = 0.38
WARNING_FRONT = 0.65
DANGER_SIDE = 0.30

TARGET_REACHED_DISTANCE = 0.45


# ============================================================
# Load V5 neural network
# ============================================================

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_ROOT / "robot_ml_model_v5.joblib"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)

FEATURES = [
    "left_sensor",
    "front_sensor",
    "right_sensor",
    "target_distance",
    "target_angle",
    "previous_action"
]

ACTIONS = [
    "FORWARD",
    "FORWARD_LEFT",
    "FORWARD_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT"
]


# ============================================================
# Utility
# ============================================================

def normalize_angle(angle):

    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def distance(x1, y1, x2, y2):

    return math.hypot(
        x2 - x1,
        y2 - y1
    )


def clamp(value, minimum, maximum):

    return max(
        minimum,
        min(value, maximum)
    )


# ============================================================
# Obstacles
# ============================================================

def point_inside_obstacle(x, y, obstacle):

    ox, oy, ow, oh = obstacle

    return (
        ox - ROBOT_RADIUS <= x <= ox + ow + ROBOT_RADIUS
        and
        oy - ROBOT_RADIUS <= y <= oy + oh + ROBOT_RADIUS
    )


def generate_obstacles():

    obstacles = []

    count = random.randint(4, 7)

    for _ in range(count):

        for _attempt in range(100):

            ow = random.uniform(
                0.5,
                1.4
            )

            oh = random.uniform(
                0.5,
                1.4
            )

            ox = random.uniform(
                0.8,
                WIDTH - ow - 0.8
            )

            oy = random.uniform(
                0.8,
                HEIGHT - oh - 0.8
            )

            obstacle = (
                ox,
                oy,
                ow,
                oh
            )

            valid = True

            for old in obstacles:

                old_x, old_y, old_w, old_h = old

                if (
                    ox < old_x + old_w + 0.25
                    and
                    ox + ow > old_x - 0.25
                    and
                    oy < old_y + old_h + 0.25
                    and
                    oy + oh > old_y - 0.25
                ):

                    valid = False
                    break

            if valid:

                obstacles.append(obstacle)
                break

    return obstacles


# ============================================================
# Ray sensor
# ============================================================

def ray_distance(
    x,
    y,
    angle,
    obstacles
):

    step = 0.04

    for d in np.arange(
        0,
        MAX_SENSOR_DISTANCE,
        step
    ):

        px = x + math.cos(angle) * d
        py = y + math.sin(angle) * d

        # World boundary
        if (
            px <= ROBOT_RADIUS
            or
            px >= WIDTH - ROBOT_RADIUS
            or
            py <= ROBOT_RADIUS
            or
            py >= HEIGHT - ROBOT_RADIUS
        ):

            return d

        for obstacle in obstacles:

            if point_inside_obstacle(
                px,
                py,
                obstacle
            ):

                return d

    return MAX_SENSOR_DISTANCE


def get_sensors(
    x,
    y,
    heading,
    obstacles
):

    left_angle = (
        heading +
        math.radians(45)
    )

    front_angle = heading

    right_angle = (
        heading -
        math.radians(45)
    )

    left = ray_distance(
        x,
        y,
        left_angle,
        obstacles
    )

    front = ray_distance(
        x,
        y,
        front_angle,
        obstacles
    )

    right = ray_distance(
        x,
        y,
        right_angle,
        obstacles
    )

    # Small sensor noise
    left += np.random.normal(
        0,
        0.05
    )

    front += np.random.normal(
        0,
        0.05
    )

    right += np.random.normal(
        0,
        0.05
    )

    left = clamp(
        left,
        0,
        MAX_SENSOR_DISTANCE
    )

    front = clamp(
        front,
        0,
        MAX_SENSOR_DISTANCE
    )

    right = clamp(
        right,
        0,
        MAX_SENSOR_DISTANCE
    )

    return left, front, right


# ============================================================
# Collision
# ============================================================

def collision(
    x,
    y,
    obstacles
):

    if (
        x <= ROBOT_RADIUS
        or
        x >= WIDTH - ROBOT_RADIUS
        or
        y <= ROBOT_RADIUS
        or
        y >= HEIGHT - ROBOT_RADIUS
    ):

        return True

    for obstacle in obstacles:

        if point_inside_obstacle(
            x,
            y,
            obstacle
        ):

            return True

    return False


# ============================================================
# Generate environment
# ============================================================

def create_environment():

    while True:

        obstacles = generate_obstacles()

        # Robot starting position
        robot_x = random.uniform(
            0.7,
            WIDTH - 0.7
        )

        robot_y = random.uniform(
            0.7,
            HEIGHT - 0.7
        )

        if any(
            point_inside_obstacle(
                robot_x,
                robot_y,
                obstacle
            )
            for obstacle in obstacles
        ):

            continue

        # Target
        target_x = random.uniform(
            0.7,
            WIDTH - 0.7
        )

        target_y = random.uniform(
            0.7,
            HEIGHT - 0.7
        )

        if any(
            point_inside_obstacle(
                target_x,
                target_y,
                obstacle
            )
            for obstacle in obstacles
        ):

            continue

        target_distance = distance(
            robot_x,
            robot_y,
            target_x,
            target_y
        )

        if target_distance < 3.0:
            continue

        return (
            obstacles,
            robot_x,
            robot_y,
            target_x,
            target_y
        )


# ============================================================
# Safety controller
# ============================================================

def safety_controller(
    action,
    left,
    front,
    right
):

    override = False

    # --------------------------------------------------------
    # Emergency front obstacle
    # --------------------------------------------------------

    if front < DANGER_FRONT:

        override = True

        if left > right:
            return "TURN_LEFT", override

        return "TURN_RIGHT", override

    # --------------------------------------------------------
    # Front warning
    # --------------------------------------------------------

    if front < WARNING_FRONT:

        override = True

        if left > right:

            return "FORWARD_LEFT", override

        return "FORWARD_RIGHT", override

    # --------------------------------------------------------
    # Side danger
    # --------------------------------------------------------

    if left < DANGER_SIDE:

        if action in [
            "FORWARD_LEFT",
            "TURN_LEFT"
        ]:

            override = True

            return "FORWARD_RIGHT", override

    if right < DANGER_SIDE:

        if action in [
            "FORWARD_RIGHT",
            "TURN_RIGHT"
        ]:

            override = True

            return "FORWARD_LEFT", override

    return action, override


# ============================================================
# Apply action
# ============================================================

def apply_action(
    action,
    x,
    y,
    heading
):

    new_heading = heading

    new_x = x
    new_y = y

    # --------------------------------------------------------
    # Pure rotation
    # --------------------------------------------------------

    if action == "TURN_LEFT":

        new_heading += TURN_SPEED

    elif action == "TURN_RIGHT":

        new_heading -= TURN_SPEED

    # --------------------------------------------------------
    # Forward-left
    # --------------------------------------------------------

    elif action == "FORWARD_LEFT":

        new_heading += TURN_SPEED

        new_x += (
            math.cos(new_heading)
            * FORWARD_SPEED
        )

        new_y += (
            math.sin(new_heading)
            * FORWARD_SPEED
        )

    # --------------------------------------------------------
    # Forward-right
    # --------------------------------------------------------

    elif action == "FORWARD_RIGHT":

        new_heading -= TURN_SPEED

        new_x += (
            math.cos(new_heading)
            * FORWARD_SPEED
        )

        new_y += (
            math.sin(new_heading)
            * FORWARD_SPEED
        )

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------

    elif action == "FORWARD":

        new_x += (
            math.cos(new_heading)
            * FORWARD_SPEED
        )

        new_y += (
            math.sin(new_heading)
            * FORWARD_SPEED
        )

    new_heading = normalize_angle(
        new_heading
    )

    return (
        new_x,
        new_y,
        new_heading
    )


# ============================================================
# Main simulator
# ============================================================

def main():

    print("=" * 65)
    print("🤖 MECORA V5 - AUTONOMOUS ROBOT")
    print("=" * 65)

    print()
    print("Controls:")
    print("L  = ML autonomous mode")
    print("M  = manual mode")
    print("R  = new random environment")
    print("ESC = exit")
    print("Arrow keys = manual movement")
    print()

    (
        obstacles,
        robot_x,
        robot_y,
        target_x,
        target_y
    ) = create_environment()

    heading = random.uniform(
        -math.pi,
        math.pi
    )

    previous_action = "FORWARD"

    mode = "ML"

    collisions = 0
    overrides = 0
    steps = 0

    target_reached = False

    path_x = [robot_x]
    path_y = [robot_y]

    fig, ax = plt.subplots(
        figsize=(11, 7)
    )

    plt.subplots_adjust(
        bottom=0.08
    )

    def reset():

        nonlocal obstacles
        nonlocal robot_x
        nonlocal robot_y
        nonlocal target_x
        nonlocal target_y
        nonlocal heading
        nonlocal previous_action
        nonlocal collisions
        nonlocal overrides
        nonlocal steps
        nonlocal target_reached
        nonlocal path_x
        nonlocal path_y

        (
            obstacles,
            robot_x,
            robot_y,
            target_x,
            target_y
        ) = create_environment()

        heading = random.uniform(
            -math.pi,
            math.pi
        )

        previous_action = "FORWARD"

        collisions = 0
        overrides = 0
        steps = 0

        target_reached = False

        path_x = [robot_x]
        path_y = [robot_y]

        print()
        print("🔄 New environment created.")

    def on_key(event):

        nonlocal mode
        nonlocal robot_x
        nonlocal robot_y
        nonlocal heading

        if event.key == "l":

            mode = "ML"

            print(
                "🧠 ML autonomous mode"
            )

        elif event.key == "m":

            mode = "MANUAL"

            print(
                "🎮 Manual mode"
            )

        elif event.key == "r":

            reset()

        elif event.key == "escape":

            plt.close()

        elif mode == "MANUAL":

            if event.key == "up":

                robot_x += (
                    math.cos(heading)
                    * FORWARD_SPEED
                )

                robot_y += (
                    math.sin(heading)
                    * FORWARD_SPEED
                )

            elif event.key == "left":

                heading += TURN_SPEED

            elif event.key == "right":

                heading -= TURN_SPEED

            elif event.key == "down":

                robot_x -= (
                    math.cos(heading)
                    * FORWARD_SPEED
                )

                robot_y -= (
                    math.sin(heading)
                    * FORWARD_SPEED
                )

    fig.canvas.mpl_connect(
        "key_press_event",
        on_key
    )

    # ========================================================
    # Animation
    # ========================================================

    def update(frame):

        nonlocal robot_x
        nonlocal robot_y
        nonlocal heading
        nonlocal previous_action
        nonlocal collisions
        nonlocal overrides
        nonlocal steps
        nonlocal target_reached

        if target_reached:

            return

        if mode == "ML":

            # ------------------------------------------------
            # Sensors
            # ------------------------------------------------

            left, front, right = get_sensors(
                robot_x,
                robot_y,
                heading,
                obstacles
            )

            # ------------------------------------------------
            # Target information
            # ------------------------------------------------

            target_direction = math.atan2(
                target_y - robot_y,
                target_x - robot_x
            )

            target_angle = normalize_angle(
                target_direction - heading
            )

            target_distance = distance(
                robot_x,
                robot_y,
                target_x,
                target_y
            )

            # ------------------------------------------------
            # Neural network
            # ------------------------------------------------

            input_data = np.array([[
                left,
                front,
                right,
                target_distance,
                target_angle,
                ACTIONS.index(
                    previous_action
                )
            ]])

            ml_action = model.predict(
                input_data
            )[0]

            # ------------------------------------------------
            # Safety controller
            # ------------------------------------------------

            action, override = safety_controller(
                ml_action,
                left,
                front,
                right
            )

            if override:

                overrides += 1

            previous_action = action

            # ------------------------------------------------
            # Apply action
            # ------------------------------------------------

            old_x = robot_x
            old_y = robot_y

            (
                robot_x,
                robot_y,
                heading
            ) = apply_action(
                action,
                robot_x,
                robot_y,
                heading
            )

            # ------------------------------------------------
            # Collision protection
            # ------------------------------------------------

            if collision(
                robot_x,
                robot_y,
                obstacles
            ):

                collisions += 1

                robot_x = old_x
                robot_y = old_y

                # Emergency escape
                if left > right:

                    heading += TURN_SPEED * 2

                else:

                    heading -= TURN_SPEED * 2

            # ------------------------------------------------
            # Keep robot inside world
            # ------------------------------------------------

            robot_x = clamp(
                robot_x,
                ROBOT_RADIUS,
                WIDTH - ROBOT_RADIUS
            )

            robot_y = clamp(
                robot_y,
                ROBOT_RADIUS,
                HEIGHT - ROBOT_RADIUS
            )

            steps += 1

            path_x.append(robot_x)
            path_y.append(robot_y)

            # ------------------------------------------------
            # Target check
            # ------------------------------------------------

            current_distance = distance(
                robot_x,
                robot_y,
                target_x,
                target_y
            )

            if (
                current_distance
                <= TARGET_REACHED_DISTANCE
            ):

                target_reached = True

                print()
                print("=" * 60)
                print("🎯 TARGET REACHED!")
                print("=" * 60)
                print(
                    f"Steps: {steps}"
                )
                print(
                    f"Collisions: {collisions}"
                )
                print(
                    f"Safety overrides: {overrides}"
                )
                print("=" * 60)

        # ====================================================
        # Draw
        # ====================================================

        ax.clear()

        ax.set_xlim(
            0,
            WIDTH
        )

        ax.set_ylim(
            0,
            HEIGHT
        )

        ax.set_aspect(
            "equal"
        )

        # ----------------------------------------------------
        # Obstacles
        # ----------------------------------------------------

        for obstacle in obstacles:

            ox, oy, ow, oh = obstacle

            ax.add_patch(
                Rectangle(
                    (ox, oy),
                    ow,
                    oh,
                    alpha=0.7
                )
            )

        # ----------------------------------------------------
        # Target
        # ----------------------------------------------------

        ax.add_patch(
            Circle(
                (
                    target_x,
                    target_y
                ),
                TARGET_RADIUS,
                fill=False,
                linewidth=2
            )
        )

        ax.text(
            target_x,
            target_y,
            "TARGET",
            ha="center",
            va="center"
        )

        # ----------------------------------------------------
        # Robot
        # ----------------------------------------------------

        ax.add_patch(
            Circle(
                (
                    robot_x,
                    robot_y
                ),
                ROBOT_RADIUS,
                fill=False,
                linewidth=2
            )
        )

        # Heading
        ax.arrow(
            robot_x,
            robot_y,
            math.cos(heading) * 0.5,
            math.sin(heading) * 0.5,
            head_width=0.12,
            length_includes_head=True
        )

        # ----------------------------------------------------
        # Path
        # ----------------------------------------------------

        if len(path_x) > 1:

            ax.plot(
                path_x,
                path_y,
                linewidth=1
            )

        # ----------------------------------------------------
        # Sensor visualization
        # ----------------------------------------------------

        left, front, right = get_sensors(
            robot_x,
            robot_y,
            heading,
            obstacles
        )

        sensor_data = [
            (
                heading + math.radians(45),
                left
            ),
            (
                heading,
                front
            ),
            (
                heading - math.radians(45),
                right
            )
        ]

        for angle, sensor_length in sensor_data:

            ax.plot(
                [
                    robot_x,
                    robot_x +
                    math.cos(angle)
                    * sensor_length
                ],
                [
                    robot_y,
                    robot_y +
                    math.sin(angle)
                    * sensor_length
                ],
                linestyle="--",
                alpha=0.5
            )

        # ----------------------------------------------------
        # Information
        # ----------------------------------------------------

        current_distance = distance(
            robot_x,
            robot_y,
            target_x,
            target_y
        )

        ax.set_title(
            "MECORA V5 | "
            f"Mode: {mode} | "
            f"Distance: {current_distance:.2f} | "
            f"Steps: {steps} | "
            f"Collisions: {collisions} | "
            f"Overrides: {overrides}"
        )

        ax.grid(
            True,
            alpha=0.2
        )

    from matplotlib.animation import FuncAnimation

    animation = FuncAnimation(
        fig,
        update,
        interval=60,
        cache_frame_data=False
    )

    plt.show()


if __name__ == "__main__":
    main()
