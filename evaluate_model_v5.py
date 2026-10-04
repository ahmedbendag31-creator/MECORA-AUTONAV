
import math
import random
import joblib
import numpy as np


# ============================================================
# MECORA V5 - Automated Navigation Benchmark
# ============================================================

WIDTH = 10.0
HEIGHT = 7.0

ROBOT_RADIUS = 0.25
TARGET_REACHED_DISTANCE = 0.45

MAX_SENSOR_DISTANCE = 5.0

FORWARD_SPEED = 0.075
TURN_SPEED = math.radians(12)

DANGER_FRONT = 0.38
WARNING_FRONT = 0.65
DANGER_SIDE = 0.30

EPISODES = 100
MAX_STEPS = 2000

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent

MODEL_FILE = PROJECT_ROOT / "robot_ml_model_v5.joblib"

if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )

model = joblib.load(MODEL_FILE)
ACTIONS = [
    "FORWARD",
    "FORWARD_LEFT",
    "FORWARD_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT"
]


# ============================================================
# Load model
# ============================================================

model = joblib.load(MODEL_FILE)


# ============================================================
# Utilities
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
# Sensors
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

    left = ray_distance(
        x,
        y,
        heading + math.radians(45),
        obstacles
    )

    front = ray_distance(
        x,
        y,
        heading,
        obstacles
    )

    right = ray_distance(
        x,
        y,
        heading - math.radians(45),
        obstacles
    )

    # Same sensor noise used during training
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
# Environment
# ============================================================

def create_environment():

    while True:

        obstacles = generate_obstacles()

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

    # Emergency obstacle
    if front < DANGER_FRONT:

        override = True

        if left > right:

            return "TURN_LEFT", override

        return "TURN_RIGHT", override

    # Warning distance
    if front < WARNING_FRONT:

        override = True

        if left > right:

            return "FORWARD_LEFT", override

        return "FORWARD_RIGHT", override

    # Left side obstacle
    if left < DANGER_SIDE:

        if action in [
            "FORWARD_LEFT",
            "TURN_LEFT"
        ]:

            override = True

            return "FORWARD_RIGHT", override

    # Right side obstacle
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

    new_x = x
    new_y = y
    new_heading = heading

    if action == "TURN_LEFT":

        new_heading += TURN_SPEED

    elif action == "TURN_RIGHT":

        new_heading -= TURN_SPEED

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
# Run one episode
# ============================================================

def run_episode():

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

    path_length = 0.0

    start_distance = distance(
        robot_x,
        robot_y,
        target_x,
        target_y
    )

    previous_x = robot_x
    previous_y = robot_y

    for step in range(
        1,
        MAX_STEPS + 1
    ):

        # ----------------------------------------------------
        # Sensors
        # ----------------------------------------------------

        left, front, right = get_sensors(
            robot_x,
            robot_y,
            heading,
            obstacles
        )

        # ----------------------------------------------------
        # Target information
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Neural network
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Safety
        # ----------------------------------------------------

        action, override = safety_controller(
            ml_action,
            left,
            front,
            right
        )

        if override:

            overrides += 1

        previous_action = action

        # ----------------------------------------------------
        # Movement
        # ----------------------------------------------------

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

        movement = distance(
            old_x,
            old_y,
            robot_x,
            robot_y
        )

        path_length += movement

        # ----------------------------------------------------
        # Collision
        # ----------------------------------------------------

        if collision(
            robot_x,
            robot_y,
            obstacles
        ):

            collisions += 1

            robot_x = old_x
            robot_y = old_y

            # Escape direction
            if left > right:

                heading += TURN_SPEED * 2

            else:

                heading -= TURN_SPEED * 2

        # ----------------------------------------------------
        # Keep inside world
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Target check
        # ----------------------------------------------------

        current_distance = distance(
            robot_x,
            robot_y,
            target_x,
            target_y
        )

        if current_distance <= TARGET_REACHED_DISTANCE:

            return {
                "success": True,
                "collisions": collisions,
                "steps": step,
                "path_length": path_length,
                "overrides": overrides,
                "start_distance": start_distance,
                "final_distance": current_distance
            }

        previous_x = robot_x
        previous_y = robot_y

    # --------------------------------------------------------
    # Timeout
    # --------------------------------------------------------

    final_distance = distance(
        robot_x,
        robot_y,
        target_x,
        target_y
    )

    return {
        "success": False,
        "collisions": collisions,
        "steps": MAX_STEPS,
        "path_length": path_length,
        "overrides": overrides,
        "start_distance": start_distance,
        "final_distance": final_distance
    }


# ============================================================
# Benchmark
# ============================================================

def main():

    print()
    print("=" * 65)
    print("🤖 MECORA V5 - 100 EPISODE BENCHMARK")
    print("=" * 65)
    print()

    results = []

    for episode in range(
        1,
        EPISODES + 1
    ):

        result = run_episode()

        results.append(result)

        status = "SUCCESS" if result["success"] else "FAILED"

        print(
            f"Episode {episode:3d}/{EPISODES} | "
            f"{status:7s} | "
            f"Steps: {result['steps']:4d} | "
            f"Distance: {result['final_distance']:.2f} | "
            f"Collisions: {result['collisions']:2d}"
        )

    # ========================================================
    # Statistics
    # ========================================================

    successful = [
        r for r in results
        if r["success"]
    ]

    failed = [
        r for r in results
        if not r["success"]
    ]

    success_rate = (
        len(successful)
        / EPISODES
        * 100
    )

    failure_rate = (
        len(failed)
        / EPISODES
        * 100
    )

    average_collisions = np.mean([
        r["collisions"]
        for r in results
    ])

    average_steps = np.mean([
        r["steps"]
        for r in results
    ])

    average_path = np.mean([
        r["path_length"]
        for r in results
    ])

    average_overrides = np.mean([
        r["overrides"]
        for r in results
    ])

    if successful:

        average_success_steps = np.mean([
            r["steps"]
            for r in successful
        ])

    else:

        average_success_steps = 0.0

    average_final_distance = np.mean([
        r["final_distance"]
        for r in results
    ])

    # ========================================================
    # Final report
    # ========================================================

    print()
    print()
    print("=" * 65)
    print("🤖 MECORA V5 FINAL BENCHMARK RESULTS")
    print("=" * 65)

    print(
        f"Total episodes:        {EPISODES}"
    )

    print(
        f"Successful episodes:   {len(successful)}"
    )

    print(
        f"Failed episodes:       {len(failed)}"
    )

    print()

    print(
        f"SUCCESS RATE:          {success_rate:.2f}%"
    )

    print(
        f"FAILURE RATE:          {failure_rate:.2f}%"
    )

    print()

    print(
        f"Average collisions:    {average_collisions:.2f}"
    )

    print(
        f"Average steps:         {average_steps:.2f}"
    )

    print(
        f"Average path length:   {average_path:.2f}"
    )

    print(
        f"Average overrides:     {average_overrides:.2f}"
    )

    print(
        f"Average final distance:{average_final_distance:.2f}"
    )

    print(
        f"Average successful steps: {average_success_steps:.2f}"
    )

    print("=" * 65)

    print()
    print("🤖 MECORA V5 benchmark complete.")


if __name__ == "__main__":
    main()
