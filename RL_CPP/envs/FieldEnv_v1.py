import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import ConvexHull
import math
from shapely.geometry import Polygon, Point, LineString
from shapely.ops import unary_union
import cv2

class FieldEnv(gym.Env):
    def __init__(self, max_size=1, num_points=8, vehicle_width=0.1, sub_steps=10, num_lidar_rays=20):
        super(FieldEnv, self).__init__()

        self.max_size = max_size
        self.num_points = num_points
        self.vehicle_width = vehicle_width
        self.sub_steps = sub_steps
        self.num_lidar_rays = num_lidar_rays

        # Normalized action space: segment distance and curvature
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float32
        )

        # Normalized observation space: num_rays with normalized distances
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_lidar_rays,),
            dtype=np.float32
        )

    def create_field(self):
        points = np.random.rand(self.num_points, 2)
        hull = ConvexHull(points)
        field_polygon = points[hull.vertices].tolist()
        self.shapely_field_poly = Polygon(field_polygon)

    def random_point_on_polygon_perimeter(self):
        # Get the coordinates and close the polygon by adding the first coordinate to the end
        coords = list(self.shapely_field_poly.exterior.coords)
        if coords[0] != coords[-1]:
            coords.append(coords[0])

        # Calculate the perimeter of the closed polygon
        perimeter = sum(LineString([coords[i], coords[i + 1]]).length for i in range(len(coords) - 1))

        # Pick a random distance along the perimeter
        random_distance = np.random.uniform(0, perimeter)
        cumulative_distance = 0

        # Find the segment that corresponds to the random distance
        for i in range(len(coords) - 1):
            line = LineString([coords[i], coords[i + 1]])
            segment_length = line.length
            
            if cumulative_distance + segment_length >= random_distance:
                segment_fraction = (random_distance - cumulative_distance) / segment_length
                px, py = line.interpolate(segment_fraction, normalized=True).xy

                # Calculate the heading perpendicular to the edge
                edge_angle_radians = math.atan2(coords[i + 1][1] - coords[i][1], coords[i + 1][0] - coords[i][0])
                perpendicular_heading_radians = edge_angle_radians + math.radians(90)
                perpendicular_heading_degrees = math.degrees(perpendicular_heading_radians)

                start_point = [px[0], py[0]]
                heading = perpendicular_heading_degrees % 360
                
                return start_point, heading

            cumulative_distance += segment_length

    def next_point_in_path(self, segment_len, segment_angle_degrees, start=False):
        x1, y1 = self.path[-1]
        total_angle_radians = math.radians(self.heading + segment_angle_degrees)

        # Calculate the endpoint
        x2 = x1 + segment_len * math.cos(total_angle_radians)
        y2 = y1 + segment_len * math.sin(total_angle_radians)

        # Calculate offsets for width
        offset_top_dx, offset_top_dy = self.calculate_offsets(total_angle_radians, self.vehicle_width)
        offset_bottom_dx, offset_bottom_dy = self.calculate_offsets(total_angle_radians, -self.vehicle_width)

        # Top and bottom endpoints
        x2_top = x2 + offset_top_dx
        y2_top = y2 + offset_top_dy
        x2_bot = x2 + offset_bottom_dx
        y2_bot = y2 + offset_bottom_dy

        if start:
            initial_angle_radians = math.radians(self.heading)
            initial_offset_top_dx, initial_offset_top_dy = self.calculate_offsets(initial_angle_radians, self.vehicle_width)
            initial_offset_bottom_dx, initial_offset_bottom_dy = self.calculate_offsets(initial_angle_radians, -self.vehicle_width)
            
            x1_top = x1 + initial_offset_top_dx
            y1_top = y1 + initial_offset_top_dy
            x1_bot = x1 + initial_offset_bottom_dx
            y1_bot = y1 + initial_offset_bottom_dy

        else:
            x1_top = y1_top = x1_bot = y1_bot = None

        return [x2, y2], [x2_top, y2_top], [x2_bot, y2_bot], [x1_top, y1_top], [x1_bot, y1_bot]

    def calculate_offsets(self, angle_radians, vehicle_width):
        offset_angle = angle_radians - math.radians(90)
        offset_dx = 0.5 * vehicle_width * math.cos(offset_angle)
        offset_dy = 0.5 * vehicle_width * math.sin(offset_angle)
        return offset_dx, offset_dy

    def steering_to_curve(self, distance, steering_angle):
        if steering_angle < -90 or steering_angle > 90:
            raise ValueError("only works with steering angles up to +/- 90 degrees")

        segment_len = distance / self.sub_steps
        segment_angle = steering_angle / self.sub_steps

        left_edge = []
        right_edge = []

        for i in range(self.sub_steps):
            if i == 0:
                mid, top, bot, initial_top, initial_bot = self.next_point_in_path(segment_len, segment_angle, start=True)
                left_edge.append(initial_top)
                right_edge.append(initial_bot)

                left_edge.append(initial_top)
                right_edge.append(initial_bot)

            else:
                mid, top, bot, _, _ = self.next_point_in_path(segment_len, segment_angle)

            self.heading = (self.heading + segment_angle) % 360

            self.path.extend([mid])
            left_edge.extend([top])
            right_edge.extend([bot])

        # Create the polygons: cover_polygon is the newly covered area, path polygon is the overall covered area
        cover_polygon = Polygon(left_edge + list(reversed(right_edge)))
        self.shapely_cover_poly = cover_polygon

        if self.shapely_path_poly is None or self.shapely_path_poly.area < 100:
            self.shapely_path_poly = self.shapely_cover_poly
        else:
            self.shapely_path_poly = unary_union([self.shapely_path_poly, self.shapely_cover_poly])

    def extend_path(self, distance, steering_angle):
        self.steering_to_curve(distance=distance, steering_angle=steering_angle)

    def calc_intersect_area(self, poly_1, poly_2):
        if poly_2.area < 150:
            intersect_area = 0
        else:
            intersect_area = poly_1.intersection(poly_2).area
        return intersect_area

    def check_if_completed(self):
        return (self.shapely_field_poly.area == self.calc_intersect_area(self.shapely_field_poly, self.shapely_path_poly))

    def get_lidar_distances(self):
        current_position = np.array(self.path[-1])
        lidar_data = []

        for i in range(self.num_lidar_rays):
            # Calculate the angle for this ray
            angle = (self.heading + i * (360 / self.num_lidar_rays)) % 360
            angle_rad = np.radians(angle)

            # Calculate the end point of the ray
            ray_end = current_position + 1000 * np.array([np.cos(angle_rad), np.sin(angle_rad)])
            ray_line = LineString([current_position, ray_end])

            # Check for intersection with polygon edges
            min_distance = 1000  # Initialize with max distance
            for j in range(len(self.shapely_field_poly.exterior.coords) - 1):
                line = LineString([self.shapely_field_poly.exterior.coords[j], self.shapely_field_poly.exterior.coords[j + 1]])
                intersection = ray_line.intersection(line)

                if intersection.is_empty:
                    continue

                if intersection.geom_type == 'Point':
                    distance = np.linalg.norm(np.array(intersection.xy) - current_position)
                elif intersection.geom_type == 'MultiPoint':
                    distance = min(np.linalg.norm(np.array(p.xy) - current_position) for p in intersection)
                else:
                    continue

                if distance < min_distance:
                    min_distance = distance

            # Store the angle and corresponding minimum distance
            lidar_data.append((angle, min_distance))

        return lidar_data

    def check_if_inside(self):
        return self.shapely_field_poly.contains(Point(self.path[-1]))

    def reset_environment(self):
        self.create_field()
        point, heading = self.random_point_on_polygon_perimeter()
        self.heading = heading
        self.path = [point]
        self.shapely_path_poly = None
        self.shapely_cover_poly = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.reset_environment()
        _, obs = zip(*self.get_lidar_distances())
        obs = np.asarray(obs, dtype=np.float32)
        info = {}  # To do: add some info
        return obs, info

    def step(self, action):
        terminated = False
        truncated = False

        steering_angle = action[0]   # Normalized Steering
        distance = action[1] 

        self.extend_path(steering_angle=steering_angle, distance=distance)

        _, obs = zip(*self.get_lidar_distances())
        obs = np.asarray(obs, dtype=np.float32)

        alpha = 1000  # Reward for new area covered
        beta = 0   # Penalty for overlap area !!! Overlap is not yet implemented
        delta = 1000 # Large reward for completing the task
        psi = 100 # Large penalty for leaving the field
        norm = abs(self.shapely_field_poly.area) # larger fields should get more reward by default

        reward = (alpha * (abs(self.shapely_cover_poly.area) / norm) * 10000) / (distance * 0.8)  # - (beta * self.overlap_area)

        if self.check_if_completed():
            reward += delta

        if not self.check_if_inside():
            reward -= psi
            terminated = True

        self.reward = reward

        return obs, reward, terminated, truncated, {}

    def render(self):
        # Create a larger blank image for super-sampling
        scale_factor = 4
        img_size = (1000 * scale_factor, 800 * scale_factor)
        img = np.ones((img_size[1], img_size[0], 3), dtype=np.uint8) * 255  # White background

        # Define scale factors to map field coordinates to image coordinates
        margin = 0.1  # 10% margin
        minx, miny, maxx, maxy = self.shapely_field_poly.bounds
        plot_bbox = [
            minx - (maxx - minx) * margin,
            maxx + (maxx - minx) * margin,
            miny - (maxy - miny) * margin,
            maxy + (maxy - miny) * margin
        ]
        scale_x = img_size[0] / (plot_bbox[1] - plot_bbox[0])
        scale_y = img_size[1] / (plot_bbox[3] - plot_bbox[2])

        def map_to_img(point):
            x = int((point[0] - plot_bbox[0]) * scale_x)
            y = img_size[1] - int((point[1] - plot_bbox[2]) * scale_y)  # Invert y-axis
            return (x, y)

        # Draw the field polygon
        field_poly = np.array([map_to_img(p) for p in list(self.shapely_field_poly.exterior.coords)], np.int32)
        cv2.fillPoly(img, [field_poly], color=(230, 250, 230))  # Light green
        cv2.polylines(img, [field_poly], isClosed=True, color=(0, 128, 0), thickness=3*scale_factor)  # Dark green border

        # Draw the path
        path_points = np.array([map_to_img(p) for p in self.path], np.int32)
        cv2.polylines(img, [path_points], isClosed=False, color=(220, 20, 60), thickness=3*scale_factor)  # Crimson path

        # Draw the path polygon
        if self.shapely_path_poly is not None:
            path_poly = np.array([map_to_img(p) for p in list(self.shapely_path_poly.exterior.coords)], np.int32)
            overlay = img.copy()
            cv2.fillPoly(overlay, [path_poly], color=(220, 20, 60))  # Crimson fill
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)  # Blend for transparency

        # Draw gridlines
        grid_color = (152, 184, 183)
        for x in np.linspace(plot_bbox[0], plot_bbox[1], 11):
            start = map_to_img((x, plot_bbox[2]))
            end = map_to_img((x, plot_bbox[3]))
            cv2.line(img, start, end, grid_color, 1*scale_factor)
        for y in np.linspace(plot_bbox[2], plot_bbox[3], 11):
            start = map_to_img((plot_bbox[0], y))
            end = map_to_img((plot_bbox[1], y))
            cv2.line(img, start, end, grid_color, 1*scale_factor)

        # Draw LIDAR rays
        lidar_data = self.get_lidar_distances()
        robot_position = map_to_img(self.path[-1])

        for angle, distance in lidar_data:
            angle_rad = np.radians(angle)
            ray_end = (
                self.path[-1][0] + distance * np.cos(angle_rad),
                self.path[-1][1] + distance * np.sin(angle_rad)
            )
            ray_end_img = map_to_img(ray_end)

            # Draw the LIDAR ray
            cv2.line(img, robot_position, ray_end_img, color=(0, 0, 255), thickness=1*scale_factor)  # Red LIDAR rays

        # Resize the image to the original size
        img = cv2.resize(img, (1000, 800), interpolation=cv2.INTER_AREA)

        # Display the image
        #cv2.imshow('Plot', img)
        #cv2.waitKey(1000)  # Wait for a key event (1ms delay)

        # Optionally save the image
        cv2.imwrite('field_map_with_lidar.png', img)

    def close(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    def draw_polygons(self, coords1, coords2):
        # Convert lists of coordinates to numpy arrays suitable for cv2.polylines
        poly1 = np.array(coords1, np.int32)
        poly1 = poly1.reshape

        poly1 = poly1.reshape((-1, 1, 2))
        
        poly2 = np.array(coords2, np.int32)
        poly2 = poly2.reshape((-1, 1, 2))

        # Create a blank image (let's assume 500x500 size with 3 color channels)
        image = np.zeros((500, 500, 3), dtype=np.uint8)

        # Draw the polygons (let's use different colors)
        cv2.polylines(image, [poly1], isClosed=True, color=(0, 255, 0), thickness=3)  # Green color
        cv2.polylines(image, [poly2], isClosed=True, color=(255, 0, 0), thickness=3)  # Blue color

        # Save the image to the specified path
        cv2.imwrite("test.png", image)

