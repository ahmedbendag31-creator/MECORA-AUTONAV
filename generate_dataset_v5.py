import random
import math
import numpy as np
import pandas as pd

# ============================================================
# MECORA V5 - Navigation-Oriented Dataset Generator
# ============================================================

WIDTH = 10.0
HEIGHT = 7.0

ROBOT_RADIUS = 0.25
MAX_SENSOR_DISTANCE = 5.0
SENSOR_NOISE = 0.05

RAW_SAMPLES = 60000

# V5 actions
ACTIONS = [
    "FORWARD",
    "FORWARD_LEFT",
    "FORWARD_RIGHT",
    "TURN_LEFT",
    "TURN_RIGHT",
]


# ============================================================
# Utility functions
# ============================================================

def clamp(value, minimum, maximum):
    return max(minimum, min(value, maximum))


def normalize_angle(angle):
    while angle > math.pi:
        angle -= 2 * math.pi

    while angle < -math.pi:
        angle += 2 * math.pi

    return angle


def distance(x1, y1, x2, y2):
    return math.hypot(x2 - x1, y2 - y1)


# ============================================================
# Random obstacle
# ============================================================

def generate_obstacles():

    obstacles = []

    number = random.randint(4, 7)

    for _ in range(number):

        for _attempt in range(100):

            width = random.uniform(0.5, 1.4)
            height = random.uniform(0.5, 1.4)

            x = random.uniform(
                0.8,
                WIDTH - width - 0.8
            )

            y = random.uniform(
                0.8,
                HEIGHT - height - 0.8
            )

            obstacle = (x, y, width, height)

            # Keep reasonable spacing between obstacles
            valid = True

            for ox, oy, ow, oh in obstacles:

                if (
                    x < ox + ow + 0.25
                    and x + width > ox - 0.25
                    and y < oy + oh + 0.25
                    and y + height > oy - 0.25
                ):
                    valid = False
                    break

            if valid:

                obstacles.append(obstacle)
                break

    return obstacles


# ============================================================
# Point inside obstacle
# ============================================================

def point_inside_obstacle(x, y, obstacle):

    ox, oy, ow, oh = obstacle

    return (
        ox - ROBOT_RADIUS <= x <= ox + ow + ROBOT_RADIUS
        and
        oy - ROBOT_RADIUS <= y <= oy + oh + ROBOT_RADIUS
    )


# ============================================================
# Ray casting sensor
# ============================================================

def ray_distance(
    x,
    y,
    angle,
    obstacles
):

    step = 0.04

    for d in np.arange(0, MAX_SENSOR_DISTANCE, step):

        px = x + math.cos(angle) * d
        py = y + math.sin(angle) * d

        # World boundary
        if (
            px <= ROBOT_RADIUS
            or px >= WIDTH - ROBOT_RADIUS
            or py <= ROBOT_RADIUS
            or py >= HEIGHT - ROBOT_RADIUS
        ):
            return d

        for obstacle in obstacles:

            if point_inside_obstacle(px, py, obstacle):
                return d

    return MAX_SENSOR_DISTANCE


# ============================================================
# Virtual sensors
# ============================================================

def get_sensors(
    x,
    y,
    heading,
    obstacles
):

    left_angle = heading + math.radians(45)
    front_angle = heading
    right_angle = heading - math.radians(45)

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

    # Sensor noise
    left += np.random.normal(0, SENSOR_NOISE)
    front += np.random.normal(0, SENSOR_NOISE)
    right += np.random.normal(0, SENSOR_NOISE)

    left = clamp(left, 0.0, MAX_SENSOR_DISTANCE)
    front = clamp(front, 0.0, MAX_SENSOR_DISTANCE)
    right = clamp(right, 0.0, MAX_SENSOR_DISTANCE)

    return left, front, right


# ============================================================
# Expert navigation controller
# ============================================================

def expert_action(
    left,
    front,
    right,
    target_angle,
    target_distance,
    previous_action
):

    # --------------------------------------------------------
    # Emergency obstacle avoidance
    # --------------------------------------------------------

    if front < 0.40:

        if left > right:
            return "TURN_LEFT"

        return "TURN_RIGHT"

    # --------------------------------------------------------
    # Strong obstacle avoidance
    # --------------------------------------------------------

    if front < 0.70:

        if left > right:
            return "FORWARD_LEFT"

        return "FORWARD_RIGHT"

    # --------------------------------------------------------
    # Side obstacle avoidance
    # --------------------------------------------------------

    if left < 0.32:

        return "FORWARD_RIGHT"

    if right < 0.32:

        return "FORWARD_LEFT"

    # --------------------------------------------------------
    # Target-oriented navigation
    # --------------------------------------------------------

    angle_deg = math.degrees(target_angle)

    # Large heading error
    if angle_deg > 35:

        return "FORWARD_LEFT"

    if angle_deg < -35:

        return "FORWARD_RIGHT"

    # Medium heading error
    if angle_deg > 12:

        return "FORWARD_LEFT"

    if angle_deg < -12:

        return "FORWARD_RIGHT"

    # --------------------------------------------------------
    # If almost aligned with target, go forward
    # --------------------------------------------------------

    return "FORWARD"


# ============================================================
# Generate one sample
# ============================================================

def generate_sample():

    obstacles = generate_obstacles()

    # --------------------------------------------------------
    # Robot position
    # --------------------------------------------------------

    for _ in range(100):

        robot_x = random.uniform(
            0.6,
            WIDTH - 0.6
        )

        robot_y = random.uniform(
            0.6,
            HEIGHT - 0.6
        )

        if not any(
            point_inside_obstacle(
                robot_x,
                robot_y,
                obstacle
            )
            for obstacle in obstacles
        ):
            break

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    for _ in range(100):

        target_x = random.uniform(
            0.6,
            WIDTH - 0.6
        )

        target_y = random.uniform(
            0.6,
            HEIGHT - 0.6
        )

        target_distance = distance(
            robot_x,
            robot_y,
            target_x,
            target_y
        )

        valid = (
            target_distance > 2.5
            and
            not any(
                point_inside_obstacle(
                    target_x,
                    target_y,
                    obstacle
                )
                for obstacle in obstacles
            )
        )

        if valid:
            break

    # --------------------------------------------------------
    # Random heading
    # --------------------------------------------------------

    heading = random.uniform(
        -math.pi,
        math.pi
    )

    # --------------------------------------------------------
    # Sensors
    # --------------------------------------------------------

    left, front, right = get_sensors(
        robot_x,
        robot_y,
        heading,
        obstacles
    )

    # --------------------------------------------------------
    # Target direction
    # --------------------------------------------------------

    target_direction = math.atan2(
        target_y - robot_y,
        target_x - robot_x
    )

    target_angle = normalize_angle(
        target_direction - heading
    )

    # --------------------------------------------------------
    # Random previous action
    # --------------------------------------------------------

    previous_action = random.choice(ACTIONS)

    # --------------------------------------------------------
    # Expert decision
    # --------------------------------------------------------

    action = expert_action(
        left,
        front,
        right,
        target_angle,
        target_distance,
        previous_action
    )

    return {
        "left_sensor": left,
        "front_sensor": front,
        "right_sensor": right,

        "target_distance": target_distance,
        "target_angle": target_angle,

        "previous_action": ACTIONS.index(
            previous_action
        ),

        "action": action
    }


# ============================================================
# Main dataset generation
# ============================================================

def main():

    print("=" * 65)
    print("🤖 MECORA V5 - DATASET GENERATION")
    print("=" * 65)

    print()
    print(f"Generating {RAW_SAMPLES} raw samples...")
    print()

    data = []

    for i in range(RAW_SAMPLES):

        sample = generate_sample()

        data.append(sample)

        if (i + 1) % 5000 == 0:

            print(
                f"Generated {i + 1}/{RAW_SAMPLES}"
            )

    df = pd.DataFrame(data)

    # ========================================================
    # Balance dataset
    # ========================================================

    print()
    print("Original action distribution:")
    print(df["action"].value_counts())

    minimum_count = df["action"].value_counts().min()

    balanced_parts = []

    for action in ACTIONS:

        action_data = df[
            df["action"] == action
        ]

        sampled = action_data.sample(
            n=minimum_count,
            random_state=42
        )

        balanced_parts.append(sampled)

    balanced_df = pd.concat(
        balanced_parts
    )

    balanced_df = balanced_df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # ========================================================
    # Save
    # ========================================================

    filename = "robot_training_data_v5.csv"

    balanced_df.to_csv(
        filename,
        index=False
    )

    print()
    print("=" * 65)
    print("V5 DATASET READY")
    print("=" * 65)

    print(
        f"Original samples: {len(df)}"
    )

    print(
        f"Balanced samples: {len(balanced_df)}"
    )

    print()
    print("Balanced action distribution:")
    print(
        balanced_df["action"].value_counts()
    )

    print()
    print(f"Saved as: {filename}")

    print()
    print("🤖 MECORA V5 dataset generation complete.")


if __name__ == "__main__":
    main()
