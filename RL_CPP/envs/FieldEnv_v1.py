import numpy as np
import gymnasium as gym
from gymnasium import spaces
from scipy.spatial import ConvexHull
import math
from shapely.geometry import Polygon, Point, LineString, MultiPoint, MultiPolygon
from shapely.ops import unary_union
from shapely import coverage_union, union, difference
import cv2
import matplotlib.pyplot as plt 
import time

# Currently we are normalizing the observation using the largest possible distance, maybe the largest distance in each env would be better 

class FieldEnv(gym.Env):
    def __init__(self, max_size=100, num_points=8, vehicle_width=1, sub_steps=10, num_lidar_rays=20, local_view_size=20, grid_size=50):
        super(FieldEnv, self).__init__()

        self.max_size = max_size
        self.num_points = num_points
        self.vehicle_width = vehicle_width
        self.sub_steps = sub_steps
        self.num_lidar_rays = num_lidar_rays
        self.lidar_data = []
        self.local_view_size = local_view_size
        self.grid_size = grid_size

        # Normalized action space: segment distance and curvature
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            shape=(2,),
            dtype=np.float32
        )

        # Normalized observation space: num_rays with normalized distances + flattened grid
        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(self.num_lidar_rays + (grid_size**2),),
            dtype=np.float32
        )

    def create_field(self):
        points = np.random.uniform(0, self.max_size, (self.num_points, 2))
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

        if self.shapely_path_poly is None:
            self.shapely_path_poly = self.shapely_cover_poly
        else:
            
            '''
            # Plot the polygon of the path 
            x, y = self.shapely_cover_poly.exterior.xy
            x2,y2 = self.shapely_path_poly.exterior.xy
            plt.figure()
            plt.plot(x, y)
            plt.plot(x2, y2)
            plt.fill(x, y, alpha=0.5, fc='red', ec='black')  # Fill the polygon with some color
            plt.show()
            '''
            
            #combined_coords = list(self.shapely_path_poly.exterior.coords) + list(cover_polygon.exterior.coords)
            #self.shapely_path_pol = Polygon(combined_coords)
            # Step 4: Create a new polygon with the combined coordinates
            
            self.shapely_path_poly = unary_union([self.shapely_path_poly, self.shapely_cover_poly])

            '''
            holes = [list(interior.coords) for interior in self.shapely_path_poly.interiors]
            if len(holes) != 0: 
                x, y = self.shapely_path_poly.exterior.xy
            
                plt.figure()
                #plt.plot(x, y)
                #plt.fill(x, y, alpha=0.5, fc='red')  # Fill the polygon with some color
                
                for hole in holes:
                    hx, hy = zip(*hole)
                    plt.plot(hx, hy, label='Hole')
                    plt.fill(hx, hy, alpha=1, fc='white')
                plt.show()
            '''

                
    def calculate_overlap(self): 
        return(self.shapely_path_poly.intersection(self.shapely_cover_poly).area)
                            

    def extend_path(self, distance, steering_angle):
        self.steering_to_curve(distance=distance, steering_angle=steering_angle)

    def calc_intersect_area(self, poly_1, poly_2):
        intersect_area = poly_1.intersection(poly_2).area
        return intersect_area

    def check_if_completed(self):
        return (self.shapely_field_poly.area == self.shapely_path_poly.area)

    def get_lidar_distances(self):
        current_position = np.array(self.path[-1])
        lidar_data = []
        norm_dist = None

        for i in range(self.num_lidar_rays):
            # Calculate the angle for this ray
            angle = (self.heading + i * (360 / self.num_lidar_rays)) % 360
            angle_rad = np.radians(angle)

            # Calculate the end point of the ray
            ray_end = current_position + (self.max_size*math.sqrt(2)) * np.array([np.cos(angle_rad), np.sin(angle_rad)])
            ray_line = LineString([current_position, ray_end])

            # Check for intersection with polygon edges
            for j in range(len(self.shapely_field_poly.exterior.coords) - 1):
                line = LineString([self.shapely_field_poly.exterior.coords[j], self.shapely_field_poly.exterior.coords[j + 1]])
                intersection = ray_line.intersection(line)

                if intersection.is_empty:
                    continue

                if isinstance(intersection, Point):
                    distance = np.linalg.norm(np.array(intersection.coords[0]) - current_position)
                
                elif isinstance(intersection, MultiPoint):
                # If there are multiple intersections, choose the closest one
                    distances = [np.linalg.norm(np.array(point.coords[0]) - current_position) for point in intersection.geoms]
                    distance = min(distances)
               
                #distance = np.linalg.norm(np.array(intersection.xy) - current_position)
                norm_dist = distance / (self.max_size*math.sqrt(2))
                
            # Store the angle and corresponding minimum distance
            lidar_data.append((angle, norm_dist))

        self.lidar_data = lidar_data
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
        _, lidar = zip(*self.get_lidar_distances())
        lidar = np.asarray(lidar, dtype=np.float32)

        obs = np.concatenate([lidar, self.get_local_coverage()])

        info = {}  # To do: add some info
        return obs, info

    def step(self, action):
        terminated = False
        truncated = False

        # The algorithm gives normalized actions (steering: +/-1 and distance: 0-1)
        steering_angle = action[0] * 60  # Steering is +/- 60 
        distance = action[1] * (self.max_size/10) + 1 # Max Segment Size is 1/10 max field size. We also have a min distance.

        self.extend_path(steering_angle=steering_angle, distance=distance)

        _, lidar = zip(*self.get_lidar_distances())
        lidar = np.asarray(lidar, dtype=np.float32)

        obs = np.concatenate([lidar, self.get_local_coverage()])

        overlap = self.calculate_overlap()

        alpha = 1.0  # Reward for new area covered
        beta = 0.5   # Penalty for overlap area
        delta = 1.0  # Reward for completing the task
        psi = 1.0    # Penalty for leaving the field
        norm_area = abs(self.shapely_field_poly.area)

        reward = (alpha * (abs(self.shapely_cover_poly.area) / norm_area))  - (beta * overlap)

        if self.check_if_completed():
            reward += delta

        if not self.check_if_inside():
            reward -= psi
            terminated = True

        self.terminated = terminated
        self.reward = reward

        return obs, reward, terminated, truncated, {}
    
    def get_local_coverage(self):
        current_pos = np.array(self.path[-1])
        heading_radians = math.radians(self.heading+180)

        half_size = self.local_view_size / 2
        cell_size = self.local_view_size / self.grid_size

        rot_matrix = np.array([
            [math.cos(heading_radians), -math.sin(heading_radians)],
            [math.sin(heading_radians), math.cos(heading_radians)]
        ])

        local_coverage = np.zeros(self.grid_size * self.grid_size, dtype=np.float32)

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                cell_center_local = np.array([
                    (i - self.grid_size / 2 + 0.5) * cell_size,
                    (j - self.grid_size / 2 + 0.5) * cell_size
                ])
                
                cell_center_rotated = np.dot(rot_matrix, cell_center_local)
                cell_center_global = current_pos + cell_center_rotated
                cell_center = Point(cell_center_global[0], cell_center_global[1])

                index = i * self.grid_size + j

                if self.shapely_field_poly.contains(cell_center):
                    if self.shapely_path_poly and self.shapely_path_poly.contains(cell_center):
                        local_coverage[index] = 1.0
                    else:
                        local_coverage[index] = 0.5
                else:
                    local_coverage[index] = 0.0

        return local_coverage

    def render(self):
        #if not self.check_if_inside():
        #    return
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

        # Draw the path
        path_points = np.array([map_to_img(p) for p in self.path], np.int32)
        cv2.polylines(img, [path_points], isClosed=False, color=(220, 20, 60), thickness=3*scale_factor)  # Crimson path

        # Draw the path polygon
        if self.shapely_path_poly is not None:
            overlay = img.copy()
            
            # Draw exterior
            path_poly = np.array([map_to_img(p) for p in list(self.shapely_path_poly.exterior.coords)], np.int32)
            cv2.fillPoly(overlay, [path_poly], color=(220, 20, 60))  # Crimson fill
            
            # Draw interiors (holes)
            for interior in self.shapely_path_poly.interiors:
                hole_poly = np.array([map_to_img(p) for p in interior.coords], np.int32)
                cv2.fillPoly(overlay, [hole_poly], color=(230, 250, 230))  # Fill holes with light green (same as field color)
            
            cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)  # Blend for transparency


        # Draw LIDAR rays if not terminated 
        if not self.terminated: 
            lidar_data = self.lidar_data
            robot_position = map_to_img(self.path[-1])

            for angle, distance in lidar_data:
                angle_rad = np.radians(angle)
                ray_end = (
                    self.path[-1][0] + (distance*(self.max_size*math.sqrt(2))) * np.cos(angle_rad),
                    self.path[-1][1] + (distance*(self.max_size*math.sqrt(2))) * np.sin(angle_rad)
                )
                ray_end_img = map_to_img(ray_end)

                # Draw the LIDAR ray
                cv2.line(img, robot_position, ray_end_img, color=(0, 0, 255), thickness=1*scale_factor)  # Red LIDAR rays


         # Draw the local coverage grid
        local_coverage = self.get_local_coverage().reshape(self.grid_size, self.grid_size)
        inset_size = 600  # Increased size of the inset
        cell_size = max(1, int(inset_size / self.grid_size))  # Adjust cell size based on new inset size
        inset = np.ones((inset_size, inset_size, 3), dtype=np.uint8) * 255

        for i in range(self.grid_size):
            for j in range(self.grid_size):
                color = (255, 255, 255)  # White for outside field
                if local_coverage[i, j] == 0.5:
                    color = (230, 250, 230)  # Light green for unvisited
                elif local_coverage[i, j] == 1.0:
                    color = (220, 20, 60)  # Crimson for visited
                
                cv2.rectangle(inset, 
                            (j*cell_size, i*cell_size), 
                            ((j+1)*cell_size, (i+1)*cell_size), 
                            color, -1)

        # Add grid lines only if cells are large enough
        if cell_size > 2:
            for i in range(0, inset_size, cell_size):
                cv2.line(inset, (i, 0), (i, inset_size), (200, 200, 200), 1)
                cv2.line(inset, (0, i), (inset_size, i), (200, 200, 200), 1)

        # Remove the title text

        # Calculate the position to place the inset (top-right corner)
        inset_position = (img.shape[1] - inset_size - 10, 10)

        # Blend the inset into the main image
        img[inset_position[1]:inset_position[1]+inset_size, 
            inset_position[0]:inset_position[0]+inset_size] = inset
        
        # Add a black border around the inset
        border_thickness = 3
        cv2.rectangle(inset, (0, 0), (inset_size-1, inset_size-1), (0, 0, 0), border_thickness)

        
        # Resize the image to the original size
        img = cv2.resize(img, (1000, 800), interpolation=cv2.INTER_AREA)

        # Optionally save the image
        cv2.imwrite('field_map_with_lidar.png', img)
        time.sleep(0.2)
        

    def close(self):
        cv2.destroyAllWindows()
        cv2.waitKey(1)

   

