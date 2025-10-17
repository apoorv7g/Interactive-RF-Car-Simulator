import pygame, sys, numpy as np, random

# ============================================================
# RL CAR SIMULATOR (Q-LEARNING)
# ------------------------------------------------------------
# Interactive environment for visualizing Q-learning applied
# to a grid-based navigation task with user-defined obstacles,
# paths, start/goal positions, and live training.
# ============================================================

# =============================
# CONFIGURATION PARAMETERS
# =============================
SCREEN_W, SCREEN_H = 1000, 600
CELL = 20
COLS = SCREEN_W // CELL
ROWS = SCREEN_H // CELL
FPS = 60

# =============================
# COLORS
# =============================
EMPTY = (240, 240, 240)
GRID_L = (200, 200, 200)
OBST = (40, 40, 40)
START_C = (50, 170, 50)
GOAL_C = (200, 60, 60)
PATH_C = (60, 140, 200)
AGENT_C = (255, 200, 0)
Q_ARROW = (0, 0, 0)

# =============================
# GRID VALUES (for logical map)
# =============================
VAL_EMPTY = 0
VAL_OBST = 1
VAL_PATH = 2
VAL_START = 3
VAL_GOAL = 4

# =============================
# INITIALIZATION
# =============================
pygame.init()
screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
pygame.display.set_caption("RL Car Simulator - Q-learning (fixed)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 18)

grid = np.zeros((ROWS, COLS), dtype=np.uint8)
start = None
goal = None

mode = "obstacle"  # Current editing mode: obstacle / route / start / goal
running_training = False

# =============================
# Q-LEARNING PARAMETERS
# =============================
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
n_states = ROWS * COLS
n_actions = len(ACTIONS)
Q = np.zeros((n_states, n_actions), dtype=np.float32)

# Hyperparameters
alpha = 0.6  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.2  # Exploration rate
min_epsilon = 0.01
eps_decay = 0.995
episode = 0
max_steps_per_episode = 1000

# Training state
agent_pos = None
prev_pos = None
state_idx = None
episode_step = 0
training_speed = 1


# ============================================================
# UTILITY FUNCTIONS
# ============================================================

def to_index(pos):
    """Convert a (row, col) position into a single integer state index."""
    y, x = pos
    return y * COLS + x


def inside(cell):
    """Return True if the given (row, col) cell lies within grid boundaries."""
    y, x = cell
    return 0 <= y < ROWS and 0 <= x < COLS


def manhattan(a, b):
    """Compute the Manhattan distance between two grid positions."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reset_episode():
    """
    Reset the environment and agent position at the start of an episode.
    Places the agent at the start cell (if defined) or the grid center.
    Resets step counters and indices.
    """
    global agent_pos, prev_pos, state_idx, episode_step
    if start is None:
        agent_pos = (ROWS // 2, COLS // 2)
    else:
        agent_pos = start
    prev_pos = agent_pos
    state_idx = to_index(agent_pos)
    episode_step = 0


def allowed(cell):
    """
    Check if the agent is allowed to move into a given cell.
    Only cells marked as route, start, or goal are valid for movement.
    """
    if not inside(cell):
        return False
    val = grid[cell]
    return val in (VAL_PATH, VAL_START, VAL_GOAL)


def step_env(action_idx):
    """
    Perform one environment step given an action index.

    Args:
        action_idx (int): Index of chosen action (0=up, 1=down, 2=left, 3=right)

    Returns:
        tuple(next_pos, reward, done)
        - next_pos: new position after applying action
        - reward: scalar reward for transition
        - done: whether the episode is over
    """
    global agent_pos
    dy, dx = ACTIONS[action_idx]
    y, x = agent_pos
    ny, nx = y + dy, x + dx

    # Check boundary
    if not inside((ny, nx)):
        return (y, x), -5, False  # hit boundary, small penalty

    cell_val = grid[ny, nx]

    # Collision with obstacle
    if cell_val == VAL_OBST:
        return (ny, nx), -100.0, True

    # Reached goal
    if (ny, nx) == goal:
        agent_pos = (ny, nx)
        return (ny, nx), 100.0, True

    # Can't move into empty cell
    if not allowed((ny, nx)):
        return (y, x), -20.0, False

    # Normal valid move
    agent_pos = (ny, nx)
    return (ny, nx), -1.0, False


def choose_action(sidx):
    """
    Epsilon-greedy action selection.

    Args:
        sidx (int): Current state index

    Returns:
        int: Chosen action index
    """
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(Q[sidx]))


def train_step():
    """
    Execute a single Q-learning training step.
    - Chooses an action using epsilon-greedy policy.
    - Applies action to environment.
    - Updates Q-table using Bellman equation.
    - Handles episode resets and epsilon decay.
    """
    global episode, epsilon, prev_pos, state_idx, episode_step
    if start is None or goal is None:
        return
    if state_idx is None:
        reset_episode()

    a = choose_action(state_idx)
    prev = agent_pos
    next_pos, r, done = step_env(a)
    next_idx = to_index(next_pos)

    # Reward shaping: encourage moving closer to goal
    d_before = manhattan(prev, goal)
    d_after = manhattan(next_pos, goal)
    if not done and r < 0:
        if d_after < d_before:
            r += 1.0
        elif d_after > d_before:
            r -= 1.0

    # Bellman update rule
    Q[state_idx, a] += alpha * (r + gamma * Q[next_idx].max() - Q[state_idx, a])

    prev_pos = prev
    state_idx = next_idx
    episode_step += 1

    # End of episode
    if done or episode_step >= max_steps_per_episode:
        episode += 1
        epsilon = max(min_epsilon, epsilon * eps_decay)
        reset_episode()


# ============================================================
# RENDERING FUNCTIONS
# ============================================================

def draw_grid():
    """Draw the full grid with color-coded cells (empty, obstacle, route, etc.)."""
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
    """Draw the agent as a colored circle in its current position."""
    if agent_pos is None:
        return
    y, x = agent_pos
    cx = x * CELL + CELL // 2
    cy = y * CELL + CELL // 2
    pygame.draw.circle(screen, AGENT_C, (cx, cy), CELL // 2 - 2)


def draw_q_arrows(scale=0.25):
    """
    Visualize Q-values by drawing arrows showing the best action per state.

    Args:
        scale (float): Relative arrow length within each cell.
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


def draw_ui():
    """Display control instructions and current training stats (episode, epsilon, etc.)."""
    lines = [
        f"Mode:{mode}  Keys: O-obst  R-route  S-start  G-goal  SPACE-train  C-clear  +/- speed",
        f"Episode:{episode}  Eps:{epsilon:.3f}  Alpha:{alpha}  Gamma:{gamma}  Steps/Frame:{training_speed}  Step:{episode_step}"
    ]
    y = 6
    for ln in lines:
        screen.blit(font.render(ln, True, (10, 10, 10)), (6, y))
        y += 20


# ============================================================
# MAIN LOOP
# ============================================================

reset_episode()

while True:
    # -------------------------
    # HANDLE INPUT EVENTS
    # -------------------------
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
        if event.type == pygame.KEYDOWN:
            k = event.key
            if k == pygame.K_o: mode = "obstacle"
            if k == pygame.K_r: mode = "route"
            if k == pygame.K_s: mode = "start"
            if k == pygame.K_g: mode = "goal"
            if k == pygame.K_SPACE:
                running_training = not running_training
                if running_training:
                    reset_episode()
            if k == pygame.K_c:
                grid.fill(VAL_EMPTY)
                start = None
                goal = None
                Q.fill(0)
                episode = 0
                epsilon = 0.2
                running_training = False
                reset_episode()
            if k in (pygame.K_EQUALS, pygame.K_PLUS):
                training_speed = min(200, training_speed + 1)
            if k in (pygame.K_MINUS, pygame.K_UNDERSCORE):
                training_speed = max(1, training_speed - 1)

        # Left-click = draw in current mode
        if pygame.mouse.get_pressed()[0]:
            mx, my = pygame.mouse.get_pos()
            cell = (my // CELL, mx // CELL)
            if inside(cell):
                r, c = cell
                if mode == "obstacle":
                    if grid[r, c] not in (VAL_START, VAL_GOAL):
                        grid[r, c] = VAL_OBST
                elif mode == "route":
                    if grid[r, c] == VAL_EMPTY:
                        grid[r, c] = VAL_PATH
                elif mode == "start":
                    if start is not None:
                        sy, sx = start
                        if grid[sy, sx] == VAL_START:
                            grid[sy, sx] = VAL_EMPTY
                    start = (r, c)
                    grid[r, c] = VAL_START
                    reset_episode()
                elif mode == "goal":
                    if goal is not None:
                        gy, gx = goal
                        if grid[gy, gx] == VAL_GOAL:
                            grid[gy, gx] = VAL_EMPTY
                    goal = (r, c)
                    grid[r, c] = VAL_GOAL

        # Right-click = erase cell
        if pygame.mouse.get_pressed()[2]:
            mx, my = pygame.mouse.get_pos()
            cell = (my // CELL, mx // CELL)
            if inside(cell):
                r, c = cell
                if grid[r, c] == VAL_START: start = None
                if grid[r, c] == VAL_GOAL: goal = None
                grid[r, c] = VAL_EMPTY

    # -------------------------
    # TRAINING UPDATE
    # -------------------------
    if running_training:
        for _ in range(training_speed):
            train_step()

    # -------------------------
    # RENDER EVERYTHING
    # -------------------------
    screen.fill((220, 220, 220))
    draw_grid()
    draw_q_arrows()
    draw_agent()

    # Outline start/goal cells
    if start:
        r, c = start
        pygame.draw.rect(screen, (0, 100, 0), pygame.Rect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4), 2)
    if goal:
        r, c = goal
        pygame.draw.rect(screen, (150, 0, 0), pygame.Rect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4), 2)

    draw_ui()
    pygame.display.flip()
    clock.tick(FPS)
