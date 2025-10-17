# ============================================================
# RL CAR SIMULATOR 
# ------------------------------------------------------------
# Interactive environment for visualizing Q-learning applied
# to a grid-based navigation task with user-defined obstacles,
# paths, start/goal positions, and live training.
# Author - Apoorv Gadiya (https://apoorv7g.github.io/portfolio/)
# Date - October 25
# ============================================================

from collections import deque
import numpy as np
import pygame
import random
import sys

# CONFIGURATION
SCREEN_W, SCREEN_H = 1100, 600
CELL = 20
COLS = SCREEN_W // CELL
ROWS = SCREEN_H // CELL
FPS = 60

# COLORS
EMPTY = (240, 240, 240)
GRID_L = (200, 200, 200)
OBST = (40, 40, 40)
START_C = (50, 170, 50)
GOAL_C = (200, 60, 60)
PATH_C = (60, 140, 200)
AGENT_C = (255, 200, 0)
Q_ARROW = (0, 0, 0)

# GRID VALUES
VAL_EMPTY = 0
VAL_OBST = 1
VAL_PATH = 2
VAL_START = 3
VAL_GOAL = 4

pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("RL Car Simulator - Q-learning")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 18)

# GRID AND AGENT STATE
grid = np.zeros((ROWS, COLS), dtype=np.uint8)
start = None
goal = None
mode = "obstacle"  # editing mode: obstacle, route, start, goal
running_training = False

# ACTIONS: (dy, dx)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
n_states = ROWS * COLS
n_actions = len(ACTIONS)
Q = np.zeros((n_states, n_actions), dtype=np.float32)

# Curiosity for exploration
visit_counts = np.zeros(n_states, dtype=np.int32)
beta = 0.5  # curiosity scaling factor

# Potential shaping
dist_to_goal = None

# Hyperparameters
alpha = 0.1
gamma = 0.95
epsilon = 0.2
min_epsilon = 0.01
eps_decay = 0.999
episode = 0
max_steps_per_episode = 1000

# Training state
agent_pos = None
prev_pos = None
state_idx = None
episode_step = 0
training_speed = 1

# Metrics
stats_rewards = []
stats_steps = []
stats_success = []
episode_reward = 0.0


# =========================================
# HELPER FUNCTIONS
# =========================================
def to_index(pos):
    """
    Convert 2D grid position to 1D Q-table index.

    Args:
        pos (tuple): (row, col)

    Returns:
        int: linear index for Q-table
    """
    y, x = pos
    return y * COLS + x


def inside(cell):
    """
    Check if the cell is inside the grid.

    Args:
        cell (tuple): (row, col)

    Returns:
        bool: True if inside grid, else False
    """
    y, x = cell
    return 0 <= y < ROWS and 0 <= x < COLS


def manhattan(a, b):
    """
    Compute Manhattan distance between two cells.

    Args:
        a (tuple): first cell (row, col)
        b (tuple): second cell (row, col)

    Returns:
        int: Manhattan distance
    """
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def compute_dist_to_goal(grid, goal):
    """
    Compute shortest distance from all reachable cells to goal using BFS.

    Args:
        grid (ndarray): grid layout
        goal (tuple): goal position (row, col)

    Returns:
        ndarray: distance array
    """
    if goal is None:
        return None
    dist = np.full((ROWS, COLS), -1, dtype=int)
    dist[goal] = 0
    q = deque([goal])
    while q:
        cy, cx = q.popleft()
        for dy, dx in ACTIONS:
            ny, nx = cy + dy, cx + dx
            npos = (ny, nx)
            if inside(npos) and dist[ny, nx] == -1 and grid[ny, nx] in (VAL_PATH, VAL_START, VAL_GOAL):
                dist[ny, nx] = dist[cy, cx] + 1
                q.append(npos)
    return dist


def reset_episode():
    """
    Reset agent position and episode state for a new episode.
    """
    global agent_pos, prev_pos, state_idx, episode_step, episode_reward
    if start is None:
        agent_pos = (ROWS // 2, COLS // 2)
    else:
        agent_pos = start
    prev_pos = agent_pos
    state_idx = to_index(agent_pos)
    episode_step = 0
    episode_reward = 0.0


def allowed(cell):
    """
    Check if a move to the cell is allowed (route, start, goal).

    Args:
        cell (tuple): (row, col)

    Returns:
        bool: True if allowed, else False
    """
    if not inside(cell):
        return False
    val = grid[cell]
    return val in (VAL_PATH, VAL_START, VAL_GOAL)


def step_env(action_idx):
    """
    Move agent according to action and return reward and done status.

    Args:
        action_idx (int): action index

    Returns:
        tuple: (new_position, reward, done)
    """
    global agent_pos
    dy, dx = ACTIONS[action_idx]
    y, x = agent_pos
    ny, nx = y + dy, x + dx

    if not inside((ny, nx)):
        return (y, x), -5, False

    cell_val = grid[ny, nx]
    if cell_val == VAL_OBST:
        return (ny, nx), -100.0, True
    if (ny, nx) == goal:
        agent_pos = (ny, nx)
        return (ny, nx), 100.0, True
    if not allowed((ny, nx)):
        return (y, x), -20.0, False

    agent_pos = (ny, nx)
    return (ny, nx), -1.0, False


def choose_action(sidx):
    """
    Select an action using epsilon-greedy policy.

    Args:
        sidx (int): state index

    Returns:
        int: chosen action index
    """
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(Q[sidx]))


def train_step():
    """
    Perform a single Q-learning training step with intrinsic curiosity
    and potential shaping rewards.
    """
    global episode, epsilon, prev_pos, state_idx, episode_step, dist_to_goal, episode_reward
    if start is None or goal is None:
        return
    if dist_to_goal is None:
        dist_to_goal = compute_dist_to_goal(grid, goal)
    if state_idx is None:
        reset_episode()

    a = choose_action(state_idx)
    prev = agent_pos
    next_pos, r, done = step_env(a)
    next_idx = to_index(next_pos)

    # Curiosity reward
    intrinsic_r = 0.0
    if not done:
        next_state_idx = to_index(next_pos)
        visit_counts[next_state_idx] += 1
        intrinsic_r = beta / np.sqrt(max(visit_counts[next_state_idx], 1))

    total_r = r + intrinsic_r

    # Potential shaping reward
    d_s = dist_to_goal[prev[0], prev[1]] if dist_to_goal is not None and dist_to_goal[prev[0], prev[1]] >= 0 else 0
    d_next = dist_to_goal[next_pos[0], next_pos[1]] if dist_to_goal is not None and dist_to_goal[next_pos[0], next_pos[1]] >= 0 else 0
    phi_s = -d_s
    phi_s_next = -d_next
    total_r += gamma * phi_s_next - phi_s

    # Q-learning update
    Q[state_idx, a] += alpha * (total_r + gamma * Q[next_idx].max() - Q[state_idx, a])

    prev_pos = prev
    state_idx = next_idx
    episode_step += 1
    episode_reward += total_r

    if done or episode_step >= max_steps_per_episode:
        success = 1 if r > 0 else 0
        stats_rewards.append(episode_reward)
        stats_steps.append(episode_step)
        stats_success.append(success)
        if len(stats_rewards) > 500:
            stats_rewards.pop(0)
            stats_steps.pop(0)
            stats_success.pop(0)
        episode += 1
        epsilon = max(min_epsilon, epsilon * eps_decay)
        episode_reward = 0.0
        reset_episode()


# =========================================
# DRAWING FUNCTIONS
# =========================================
def draw_grid():
    """
    Draw the grid, obstacles, paths, start, and goal.
    """
    for r in range(ROWS):
        for c in range(COLS):
            val = grid[r, c]
            rect = pygame.Rect(c * CELL, r * CELL, CELL, CELL)
            if val == VAL_EMPTY:
                color = EMPTY
            elif val == VAL_OBST:
                color = OBST
            elif val == VAL_PATH:
                color = PATH_C
            elif val == VAL_START:
                color = START_C
            elif val == VAL_GOAL:
                color = GOAL_C
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, GRID_L, rect, 1)


def draw_agent():
    """
    Draw the agent as a circle in its current position.
    """
    if agent_pos is None:
        return
    y, x = agent_pos
    cx = x * CELL + CELL // 2
    cy = y * CELL + CELL // 2
    pygame.draw.circle(screen, AGENT_C, (cx, cy), CELL // 2 - 2)


def draw_q_arrows(scale=0.25):
    """
    Draw arrows representing the best action Q-values for each cell.
    """
    for r in range(ROWS):
        for c in range(COLS):
            if grid[r, c] == VAL_OBST:
                continue
            s = to_index((r, c))
            best_a = int(np.argmax(Q[s]))
            qv = Q[s, best_a]
            if qv <= 0:
                continue
            cx = c * CELL + CELL // 2
            cy = r * CELL + CELL // 2
            dy, dx = ACTIONS[best_a]
            ex = cx + int(dx * CELL * scale)
            ey = cy + int(dy * CELL * scale)
            pygame.draw.line(screen, Q_ARROW, (cx, cy), (ex, ey), 2)
            pygame.draw.circle(screen, Q_ARROW, (ex, ey), 3)


def draw_stats():
    """
    Draw statistics for rewards, steps, and success rates.
    """
    base_x = SCREEN_W - 250
    base_y = 50
    width = 160
    height = 100

    if len(stats_rewards) < 2:
        return

    max_r = max(stats_rewards)
    min_r = min(stats_rewards)
    max_s = max(stats_steps)

    def norm(val, minv, maxv):
        return 0 if maxv == minv else (val - minv) / (maxv - minv)

    # Reward graph
    pygame.draw.rect(screen, (230, 230, 230), (base_x, base_y, width, height))
    pygame.draw.rect(screen, (200, 200, 200), (base_x, base_y, width, height), 1)
    for i in range(1, len(stats_rewards)):
        x1 = base_x + (i - 1) * width / (len(stats_rewards) - 1)
        x2 = base_x + i * width / (len(stats_rewards) - 1)
        y1 = base_y + height - norm(stats_rewards[i - 1], min_r, max_r) * height
        y2 = base_y + height - norm(stats_rewards[i], min_r, max_r) * height
        pygame.draw.line(screen, (0, 150, 0), (x1, y1), (x2, y2), 2)
    screen.blit(font.render("Rewards", True, (0, 0, 0)), (base_x + 50, base_y - 18))

    # Steps graph
    base_y2 = base_y + height + 30
    pygame.draw.rect(screen, (230, 230, 230), (base_x, base_y2, width, height))
    pygame.draw.rect(screen, (200, 200, 200), (base_x, base_y2, width, height), 1)
    for i in range(1, len(stats_steps)):
        x1 = base_x + (i - 1) * width / (len(stats_steps) - 1)
        x2 = base_x + i * width / (len(stats_steps) - 1)
        y1 = base_y2 + height - norm(stats_steps[i - 1], 0, max_s) * height
        y2 = base_y2 + height - norm(stats_steps[i], 0, max_s) * height
        pygame.draw.line(screen, (150, 0, 0), (x1, y1), (x2, y2), 2)
    screen.blit(font.render("Steps", True, (0, 0, 0)), (base_x + 60, base_y2 - 18))

    # Success bar
    recent_n = min(50, len(stats_success))
    if recent_n > 0:
        success_rate = sum(stats_success[-recent_n:]) / recent_n
        base_y3 = base_y2 + height + 30
        pygame.draw.rect(screen, (230, 230, 230), (base_x, base_y3, width, 20))
        pygame.draw.rect(screen, (200, 200, 200), (base_x, base_y3, width, 20), 1)
        bar_width = int(success_rate * width)
        pygame.draw.rect(screen, (0, 150, 0), (base_x, base_y3, bar_width, 20))
        text = font.render(f"Success: {success_rate:.2f}", True, (0, 0, 0))
        screen.blit(text, (base_x + 5, base_y3 - 18))

    # Avg reward and steps
    if len(stats_rewards) > 0:
        avg_reward = np.mean(stats_rewards[-100:]) if len(stats_rewards) > 100 else np.mean(stats_rewards)
        avg_steps = np.mean(stats_steps[-100:]) if len(stats_steps) > 100 else np.mean(stats_steps)
        avg_text = font.render(f"Avg R: {avg_reward:.1f} | Avg S: {avg_steps:.0f}", True, (0, 0, 0))
        screen.blit(avg_text, (base_x, base_y3 + 25))


def draw_ui():
    """
    Draw informational UI: mode, keys, episode, epsilon, alpha, gamma, speed.
    """
    lines = [
        f"Mode:{mode}  Keys: O-obst  R-route  S-start  G-goal  SPACE-train  C-clear  +/- speed",
        f"Episode:{episode}  Eps:{epsilon:.3f}  Alpha:{alpha}  Gamma:{gamma}  Steps/Frame:{training_speed}  Step:{episode_step}"
    ]
    y = 6
    for ln in lines:
        screen.blit(font.render(ln, True, (10, 10, 10)), (6, y))
        y += 20


# =========================================
# MAIN LOOP
# =========================================
reset_episode()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()

        # Keyboard input handling
        if event.type == pygame.KEYDOWN:
            k = event.key
            if k == pygame.K_o:
                mode = "obstacle"
            if k == pygame.K_r:
                mode = "route"
            if k == pygame.K_s:
                mode = "start"
            if k == pygame.K_g:
                mode = "goal"
            if k == pygame.K_SPACE:
                running_training = not running_training
                if running_training:
                    reset_episode()
                    dist_to_goal = None
            if k == pygame.K_c:
                # Clear grid
                grid.fill(VAL_EMPTY)
                start = None
                goal = None
                Q.fill(0)
                episode = 0
                epsilon = 0.2
                running_training = False
                reset_episode()
                visit_counts.fill(0)
                dist_to_goal = None
                stats_rewards.clear()
                stats_steps.clear()
                stats_success.clear()
            if k == pygame.K_EQUALS or k == pygame.K_PLUS:
                training_speed = min(200, training_speed + 1)
            if k == pygame.K_MINUS or k == pygame.K_UNDERSCORE:
                training_speed = max(1, training_speed - 1)

        # Mouse input handling
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            cell = (my // CELL, mx // CELL)
            if inside(cell):
                r, c = cell
                old_mode = mode
                if mode == "obstacle" and grid[r, c] not in (VAL_START, VAL_GOAL):
                    grid[r, c] = VAL_OBST
                elif mode == "route" and grid[r, c] == VAL_EMPTY:
                    grid[r, c] = VAL_PATH
                elif mode == "start":
                    if start is not None:
                        sy, sx = start
                        if grid[sy, sx] == VAL_START:
                            grid[sy, sx] = VAL_EMPTY
                    start = (r, c)
                    grid[r, c] = VAL_START
                    reset_episode()
                    dist_to_goal = None
                elif mode == "goal":
                    if goal is not None:
                        gy, gx = goal
                        if grid[gy, gx] == VAL_GOAL:
                            grid[gy, gx] = VAL_EMPTY
                    goal = (r, c)
                    grid[r, c] = VAL_GOAL
                    dist_to_goal = None
                if old_mode in ("start", "goal"):
                    dist_to_goal = None

        if pygame.mouse.get_pressed()[2]:
            mx, my = pygame.mouse.get_pos()
            cell = (my // CELL, mx // CELL)
            if inside(cell):
                r, c = cell
                if grid[r, c] == VAL_START:
                    start = None
                    dist_to_goal = None
                if grid[r, c] == VAL_GOAL:
                    goal = None
                    dist_to_goal = None
                grid[r, c] = VAL_EMPTY
                if running_training:
                    dist_to_goal = None

    # Training step
    if running_training:
        for _ in range(training_speed):
            train_step()

    # Rendering
    screen.fill((220, 220, 220))
    draw_grid()
    draw_q_arrows()
    draw_agent()
    if start:
        r, c = start
        pygame.draw.rect(screen, (0, 100, 0), pygame.Rect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4), 2)
    if goal:
        r, c = goal
        pygame.draw.rect(screen, (150, 0, 0), pygame.Rect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4), 2)
    draw_ui()
    draw_stats()
    pygame.display.flip()
    clock.tick(FPS)
