# ============================================================
# RL CAR SIMULATOR 
# ------------------------------------------------------------
# Interactive environment for visualizing Q-learning applied
# to a grid-based navigation task with user-defined obstacles,
# paths, start/goal positions, and live training.
# Author - Apoorv Gadiya (https://apoorv7g.github.io/portfolio/)
# Date - October 25
# ============================================================

import pygame, sys, numpy as np, random

# CONFIG
SCREEN_W, SCREEN_H = 1000, 600
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
pygame.display.set_caption("RL Car Simulator - Q-learning (fixed)")
clock = pygame.time.Clock()
font = pygame.font.SysFont("Consolas", 18)

grid = np.zeros((ROWS, COLS), dtype=np.uint8)
start = None
goal = None

mode = "obstacle"  # obstacle, route, start, goal
running_training = False

# ACTIONS: (dy,dx)
ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
n_states = ROWS * COLS
n_actions = len(ACTIONS)
Q = np.zeros((n_states, n_actions), dtype=np.float32)

# Curiosity: visit counts
visit_counts = np.zeros(n_states, dtype=np.int32)
beta = 0.1  # Curiosity scaling factor; tune as needed (0.05-0.2)

# Hyperparams
alpha = 0.6
gamma = 0.95
epsilon = 0.2
min_epsilon = 0.01
eps_decay = 0.995
episode = 0
max_steps_per_episode = 1000

# Training persistent episode state
agent_pos = None
prev_pos = None
state_idx = None
episode_step = 0
training_speed = 1


def to_index(pos):
    y, x = pos
    return y * COLS + x


def inside(cell):
    y, x = cell
    return 0 <= y < ROWS and 0 <= x < COLS


def manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def reset_episode():
    global agent_pos, prev_pos, state_idx, episode_step
    if start is None:
        agent_pos = (ROWS // 2, COLS // 2)
    else:
        agent_pos = start
    prev_pos = agent_pos
    state_idx = to_index(agent_pos)
    episode_step = 0


def allowed(cell):
    # car must stay on route cells or start/goal
    if not inside(cell): return False
    val = grid[cell]
    return val in (VAL_PATH, VAL_START, VAL_GOAL)


def step_env(action_idx):
    global agent_pos
    dy, dx = ACTIONS[action_idx]
    y, x = agent_pos
    ny, nx = y + dy, x + dx
    if not inside((ny, nx)):
        # outside bounds: blocked and penalize
        return (y, x), -5, False
    cell_val = grid[ny, nx]
    if cell_val == VAL_OBST:
        # collision terminal
        return (ny, nx), -100.0, True
    if (ny, nx) == goal:
        agent_pos = (ny, nx)
        return (ny, nx), 100.0, True
    # enforce route constraint: cannot move into empty cells
    if not allowed((ny, nx)):
        # blocked move: stay in place and give penalty
        return (y, x), -20.0, False
    # normal move along route
    agent_pos = (ny, nx)
    return (ny, nx), -1.0, False


def choose_action(sidx):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    return int(np.argmax(Q[sidx]))


def train_step():
    global episode, epsilon, prev_pos, state_idx, episode_step
    if start is None or goal is None:
        return
    if state_idx is None:
        reset_episode()
    a = choose_action(state_idx)
    prev = agent_pos
    next_pos, r, done = step_env(a)
    next_idx = to_index(next_pos)
    # Intrinsic curiosity reward (added on state entry)
    intrinsic_r = 0.0
    if not done:
        next_state_idx = to_index(next_pos)
        visit_counts[next_state_idx] += 1  # Increment on entry
        intrinsic_r = beta / np.sqrt(max(visit_counts[next_state_idx], 1))  # Avoid div-by-zero
    total_r = r + intrinsic_r
    # Potential-based shaping: Phi(s) = -manhattan(s, goal)
    phi_s = -manhattan(prev, goal)
    phi_s_next = -manhattan(next_pos, goal)
    total_r += gamma * phi_s_next - phi_s
    # Q update uses previous state & action
    Q[state_idx, a] += alpha * (total_r + gamma * Q[next_idx].max() - Q[state_idx, a])
    prev_pos = prev
    state_idx = next_idx
    episode_step += 1
    if done or episode_step >= max_steps_per_episode:
        episode += 1
        epsilon = max(min_epsilon, epsilon * eps_decay)
        # reset episode state for next episode
        reset_episode()


def draw_grid():
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
    if agent_pos is None:
        return
    y, x = agent_pos
    cx = x * CELL + CELL // 2
    cy = y * CELL + CELL // 2
    pygame.draw.circle(screen, AGENT_C, (cx, cy), CELL // 2 - 2)


def draw_q_arrows(scale=0.25):
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
    lines = [
        f"Mode:{mode}  Keys: O-obst  R-route  S-start  G-goal  SPACE-train  C-clear  +/- speed",
        f"Episode:{episode}  Eps:{epsilon:.3f}  Alpha:{alpha}  Gamma:{gamma}  Steps/Frame:{training_speed}  Step:{episode_step}"
    ]
    y = 6
    for ln in lines:
        screen.blit(font.render(ln, True, (10, 10, 10)), (6, y))
        y += 20


# initialize
reset_episode()

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            sys.exit()
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
            if k == pygame.K_c:
                grid.fill(VAL_EMPTY)
                start = None
                goal = None
                Q.fill(0)
                episode = 0
                epsilon = 0.2
                running_training = False
                reset_episode()
                visit_counts.fill(0)  # Reset visits on clear
            if k == pygame.K_EQUALS or k == pygame.K_PLUS:
                training_speed = min(200, training_speed + 1)
            if k == pygame.K_MINUS or k == pygame.K_UNDERSCORE:
                training_speed = max(1, training_speed - 1)
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
                        if grid[sy, sx] == VAL_START: grid[sy, sx] = VAL_EMPTY
                    start = (r, c)
                    grid[r, c] = VAL_START
                    reset_episode()
                elif mode == "goal":
                    if goal is not None:
                        gy, gx = goal
                        if grid[gy, gx] == VAL_GOAL: grid[gy, gx] = VAL_EMPTY
                    goal = (r, c)
                    grid[r, c] = VAL_GOAL
        if pygame.mouse.get_pressed()[2]:
            mx, my = pygame.mouse.get_pos()
            cell = (my // CELL, mx // CELL)
            if inside(cell):
                r, c = cell
                if grid[r, c] == VAL_START: start = None
                if grid[r, c] == VAL_GOAL: goal = None
                grid[r, c] = VAL_EMPTY

    if running_training:
        for _ in range(training_speed):
            train_step()

    # render
    screen.fill((220, 220, 220))
    draw_grid()
    draw_q_arrows()
    draw_agent()
    # outlines
    if start:
        r, c = start
        pygame.draw.rect(screen, (0, 100, 0), pygame.Rect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4), 2)
    if goal:
        r, c = goal
        pygame.draw.rect(screen, (150, 0, 0), pygame.Rect(c * CELL + 2, r * CELL + 2, CELL - 4, CELL - 4), 2)
    draw_ui()
    pygame.display.flip()
    clock.tick(FPS)
