import gymnasium as gym 
from gymnasium import Wrapper
import matplotlib.pyplot as plt 
from matplotlib.patches import Polygon
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

class VisualizeWrapper(Wrapper): 
    def __init__(self, env):
        super().__init__(env)
        self.path_line = None
        self.path_polygon = None
        

        #self.fig = plt.figure(figsize=(8, 8))  # Larger figure size
        #self.ax = self.fig.gca()
        #plt.ion()  # Turn on interactive mode
        
    def reset(self):
        obs = self.env.reset()
        self.fig = plt.figure(figsize=(8, 8))  # Larger figure size
        self.ax = self.fig.gca()
        plt.ion()  # Turn on interactive mode
        field_polygon = Polygon(self.env.unwrapped.playground.polygon, facecolor='lightgreen', edgecolor='green', alpha=0.5)
        self.ax.add_patch(field_polygon)
        self.ax.set_xlim(self.env.unwrapped.playground.bounding_box[0], self.env.unwrapped.playground.bounding_box[1])
        self.ax.set_ylim(self.env.unwrapped.playground.bounding_box[2], self.env.unwrapped.playground.bounding_box[3])
        self.ax.set_aspect('equal', 'box')
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        #self.ax.legend()

        # Remove spines for a cleaner look
        for spine in self.ax.spines.values():
            spine.set_visible(False)

        #self._update_display()
        return obs
    
    def step(self, action):
        obs, reward, terminated, truncated, info  = self.env.step(action)
        self._update_display()
        return obs, reward, terminated, truncated, info
    
    def _update_display(self):
        if self.path_line:
            self.path_line.remove()
        if self.path_polygon:
            self.path_polygon.remove()
        # Plot the path if provided
        if self.env.unwrapped.playground.path:
            path_x, path_y = zip(*self.env.unwrapped.playground.path)
            self.path_line = Line2D(path_x, path_y, color='r', linewidth=2, label='Path')
            self.ax.add_line(self.path_line)

        # Plot the path polygon if provided
        if self.env.unwrapped.playground.path_polygon:
            self.path_polygon = Polygon(self.env.unwrapped.playground.path_polygon, facecolor='red', edgecolor='red', alpha=0.3)
            self.ax.add_patch(self.path_polygon)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(0.1)

        
    

