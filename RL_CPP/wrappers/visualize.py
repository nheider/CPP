import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import gymnasium as gym
from gymnasium import Wrapper
import numpy as np
from PIL import Image
import io

class VisualizeWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.fig = None
        self.ax = None
        self.path_line = None
        self.path_polygon = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self._setup_plot()
        return obs, info

    def _setup_plot(self):
        if self.fig is None:
            self.fig, self.ax = plt.subplots(figsize=(8, 8))
        else:
            self.ax.clear()

        field_polygon = Polygon(self.env.unwrapped.polygon, facecolor='lightgreen', edgecolor='green', alpha=0.5)
        self.ax.add_patch(field_polygon)
        self.ax.set_xlim(self.env.unwrapped.bounding_box[0], self.env.unwrapped.bounding_box[1])
        self.ax.set_ylim(self.env.unwrapped.bounding_box[2], self.env.unwrapped.bounding_box[3])
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')

        for spine in self.ax.spines.values():
            spine.set_visible(False)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._update_display()
        return obs, reward, terminated, truncated, info

    def _update_display(self):
        if self.path_line:
            self.path_line.remove()
        if self.path_polygon:
            self.path_polygon.remove()

        if self.env.unwrapped.path:
            path_x, path_y = zip(*self.env.unwrapped.path)
            self.path_line = Line2D(path_x, path_y, color='r', linewidth=2, label='Path')
            self.ax.add_line(self.path_line)

        if self.env.unwrapped.path_polygon:
            self.path_polygon = Polygon(self.env.unwrapped.path_polygon, facecolor='red', edgecolor='red', alpha=0.3)
            self.ax.add_patch(self.path_polygon)

    '''
    def render(self):
        buf = io.BytesIO()
        self.fig.canvas.print_png(buf)
        buf.seek(0)
        img = Image.open(buf)
        return np.array(img)
    '''

    def close(self):
        if self.fig:
            plt.close(self.fig)
        self.fig = None
        self.ax = None