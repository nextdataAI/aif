import numpy as np
import random


def create_maze(height, width):
    # Create a grid filled with walls
    maze = np.full((height * 2 + 1, width * 2 + 1), chr(32), dtype=str)

    # Define the starting point
    x, y = (1, 1)
    maze[x, y] = chr(46)  # Open space

    # Initialize the stack with the starting point
    stack = [(x, y)]
    while len(stack) > 0:
        x, y = stack[-1]

        # Define possible directions
        directions = [(0, 2), (2, 0), (0, -2), (-2, 0)]
        random.shuffle(directions)

        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if 1 <= nx < height * 2 and 1 <= ny < width * 2 and maze[nx, ny] == chr(32):
                maze[nx, ny] = chr(46)  # Open space
                maze[x + dx // 2, y + dy // 2] = chr(46)  # Open space
                stack.append((nx, ny))
                break
        else:
            stack.pop()

    # Find all open spaces
    open_spaces = [(i, j) for i in range(height * 2 + 1) for j in range(width * 2 + 1) if maze[i, j] == chr(46)]

    # Randomly select a position for the player and the target
    player_pos = random.choice(open_spaces)
    open_spaces.remove(player_pos)
    target_pos = random.choice(open_spaces)

    # Place the player and the target
    maze[player_pos[0], player_pos[1]] = chr(64)  # Player
    maze[target_pos[0], target_pos[1]] = chr(60)  # Target

    strings = []
    my_maze = ''
    for elem in maze:
        strings.append(''.join(elem))
    for string in create_maze(5, 21):
        my_maze += string + '\n'
    return my_maze


if __name__ == '__main__':
    print(create_maze(5, 21))