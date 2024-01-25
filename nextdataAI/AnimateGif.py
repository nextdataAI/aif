import gym
import numpy as np
from .utils import get_player_location, get_target_location
from tqdm import tqdm
import imageio.v2 as imageio
from PIL import Image
from data.dataset import get_player_floor_target_png
import os

__all__ = ["Animator"]


class Animator:
    def __init__(self, size, game_map, path, visited, file_path=None, fps=30, disable_line=True):
        self.game_map = game_map.get('pixel').copy()
        self.chars = game_map.get('chars').copy()
        self.start = get_player_location(self.chars)
        self.fps = fps
        self.target = get_target_location(self.chars)
        self.visited = visited
        self.path = path
        self.disable_line = disable_line
        if self.target in path:
            self.path.remove(self.target)
        self.visited_actions = self.coordinates_to_action(visited)
        self.path_actions = self.coordinates_to_action(path)
        self.done = []
        self.size = size
        if file_path is not None:
            self.file_path = 'TestsAnimations/' + file_path
        self.player, self.floor, self.floor_red, self.floor_yellow, self.flood_blue, self.target_img = get_player_floor_target_png(
            True)

    @staticmethod
    def tmp_folder():
        if not os.path.exists('tmp'):
            os.mkdir('tmp')
        else:
            for file in os.listdir('tmp'):
                os.remove(f'tmp/{file}')

    @staticmethod
    def coordinates_to_action(coordinates_array: list) -> list:
        """
        Convert coordinates to action.
        Args:
            coordinates_array (np.ndarray): The coordinates array.
        Returns:
            int: The action.
        """
        out = []
        coordinates_array = coordinates_array.copy() if isinstance(coordinates_array, list) else list(coordinates_array)
        starting = coordinates_array.pop(0)
        for i in range(len(coordinates_array)):
            if starting[1] - coordinates_array[i][1] == 1:
                out.append({coordinates_array[i]: 3})
            elif starting[1] - coordinates_array[i][1] == -1:
                out.append({coordinates_array[i]: 1})
            elif starting[0] - coordinates_array[i][0] == 1:
                out.append({coordinates_array[i]: 0})
            elif starting[0] - coordinates_array[i][0] == -1:
                out.append({coordinates_array[i]: 2})
            starting = coordinates_array[i]
        return out

    def __call__(self):
        self.tmp_folder()
        new_game_map, out = self.__print_visited(self.game_map, self.visited)
        self.__animate__(new_game_map, self.path_actions, images=out)
        self.tmp_folder()

    def __animate__(self, game_map, visited_actions, images):
        """
        Animate the path found by the algorithm.
        Args:
            game_map: The game map.
            path: The path found by the algorithm.
            visited: The visited nodes.
        """
        # create the animation
        animation = images
        local_game_map = game_map.copy()
        counter = int(list(images[-1].keys())[-1]) + 1
        for entry in tqdm(visited_actions, disable=self.disable_line):
            state = tuple(entry.keys())[0]
            action = entry[state]
            # get the current state
            # move the player in the game map
            local_game_map = self.__move_player__(local_game_map.copy(), action, counter)
            # add the game map to the animation
            animation.append({counter: imageio.imread(f'tmp/animation_{counter}.png')})
            self.done.append(state)
            counter += 1
        # save the animation
        out = [list(elem.values())[-1] for elem in animation]
        imageio.mimsave(f'{self.file_path}', out, fps=self.fps)

    def __move_player__(self, game_map, action, counter=0):
        """
        Move the player in a direction.
        Args:
            state (State): The state in which to move the player.
            game_map: The game map.
        Returns:
            new_game_map: The new game map.
        """
        tmp_image = game_map.copy()
        player_img = self.player.copy()
        floor_img = self.flood_blue.copy()
        # move the player
        for i in range(0, 336, 16):
            for j in range(0, 1264, 16):
                tmp2 = tmp_image[i:i + 16, j:j + 16].reshape(-1)
                tmp3 = player_img.reshape(-1)
                if np.array_equal(tmp2, tmp3):
                    tmp_image[i:i + 16, j:j + 16] = floor_img[:, :, :3]
                    if action == 0:
                        tmp_image[i - 16:i, j:j + 16] = player_img
                    elif action == 2:
                        tmp_image[i + 16:i + 32, j:j + 16] = player_img
                    elif action == 3:
                        tmp_image[i:i + 16, j - 16:j] = player_img
                    elif action == 1:
                        tmp_image[i:i + 16, j + 16:j + 32] = player_img
                    tmp_image2 = self.crop(tmp_image)
                    Image.fromarray(tmp_image2).save(f'tmp/animation_{counter}.png')
                    return tmp_image

    def __print_visited(self, game_map, visited):
        """
        Print the visited nodes.
        Args:
            game_map: The game map.
            visited: The visited nodes.
        """
        tmp_image = game_map.copy()
        counter = 0
        out = []
        for state in visited:
            floor_img = self.floor_red.copy() if state in self.done else self.floor_yellow.copy()
            tmp1 = tmp_image[state[0] * 16:state[0] * 16 + 16, state[1] * 16:state[1] * 16 + 16]
            if not np.array_equal(tmp1, self.player) and not np.array_equal(tmp1, self.target_img):
                tmp_image[state[0] * 16:state[0] * 16 + 16, state[1] * 16:state[1] * 16 + 16] = floor_img[:, :, :3]
                tmp_image2 = self.crop(tmp_image)
                Image.fromarray(tmp_image2).save(f'tmp/animation_{counter}.png')
                out.append({counter: imageio.imread(f'tmp/animation_{counter}.png')})
                counter += 1
        return tmp_image, out

    def crop(self, image):
        """
        Crop the image.
        Args:
            image: The image to crop.
        """
        if self.size == 'small':
            return image[112:224, 576:688]
        elif self.size == 'medium':
            return image[80:288, 512:720]
        elif self.size == 'large':
            return image[48:336, 276:990]