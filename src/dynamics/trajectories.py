import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple
from src.utils.tools import ssa

class TrajectoryStrategy(ABC):
    @abstractmethod
    def compute_velocity_commands(self, current_state, dt: float) -> Tuple[float, float]:
        """
        Returns (target_speed, target_yaw_rate) based on current state.
        """
        pass

class ConstantVelocityTrajectory(TrajectoryStrategy):
    """Original behavior: maintain initial velocity, zero turn rate (plus noise)"""
    def compute_velocity_commands(self, current_state, dt: float) -> Tuple[float, float]:
        speed = np.hypot(current_state.vel_x, current_state.vel_y)
        return speed, 0.0

class CircleTrajectory(TrajectoryStrategy):
    """
    Orbits a specific point (center_x, center_y).
    This ensures the boat shows different aspects to the sensor.
    """
    def __init__(self, center: Tuple[float, float], target_speed: float, radius: float = None, clockwise: bool = True):
        self.center = np.array(center)
        self.target_speed = target_speed
        self.clockwise = clockwise
        self.radius = radius # If None, maintains current radius
        
        # Simple P-controller gain for radius correction
        self.Kp = 0.5 

    def compute_velocity_commands(self, current_state, dt: float) -> Tuple[float, float]:
        # Ship position relative to Center:
        dp = np.array([current_state.x, current_state.y]) - self.center
        dist = np.linalg.norm(dp)
        angle_to_ship = np.arctan2(dp[1], dp[0]) 

        offset = np.pi/2 if self.clockwise else -np.pi/2
        desired_heading = ssa(angle_to_ship + offset)
        
        # Radial correction: If we drifted out, turn in slightly
        if self.radius is not None:
            err_r = dist - self.radius
            # Correction angle
            correction = np.arctan(self.Kp * err_r) * (1 if self.clockwise else -1)
            desired_heading = ssa(desired_heading + correction)

        # Calculate Yaw Rate to achieve desired heading
        # Simple P-controller for steering
        heading_error = ssa(desired_heading - current_state.yaw)
        target_yaw_rate = 2.0 * heading_error 

        return self.target_speed, target_yaw_rate

class WaypointTrajectory(TrajectoryStrategy):
    """Follows a list of (x, y) waypoints using Line-of-Sight."""
    def __init__(self, waypoints: List[Tuple[float, float]], target_speed: float, acceptance_radius: float = 5.0):
        self.waypoints = waypoints
        self.target_speed = target_speed
        self.acceptance_radius = acceptance_radius
        self.current_idx = 0
        self.loop = True

    def compute_velocity_commands(self, current_state, dt: float) -> Tuple[float, float]:
        if self.current_idx >= len(self.waypoints):
            return 0.0, 0.0 # Stop

        target = np.array(self.waypoints[self.current_idx])
        pos = np.array([current_state.x, current_state.y])
        
        dist = np.linalg.norm(target - pos)
        
        # Check if reached
        if dist < self.acceptance_radius:
            self.current_idx += 1
            if self.loop and self.current_idx >= len(self.waypoints):
                self.current_idx = 0
            return self.compute_velocity_commands(current_state, dt)

        # LOS Guidance
        desired_heading = np.arctan2(target[1] - pos[1], target[0] - pos[0])
        heading_error = ssa(desired_heading - current_state.yaw)
        
        # P-Controller for yaw rate
        target_yaw_rate = 1.5 * heading_error
        
        return self.target_speed, target_yaw_rate

class DynamicWaypointTrajectory(TrajectoryStrategy):
    """
    Follows a list of (x, y, speed) waypoints using Line-of-Sight.
    Accelerates/decelerates to reach the target speed for each segment.
    """
    def __init__(self, waypoints: List[Tuple[float, float, float]], initial_speed: float, acceptance_radius: float = 5.0):
        self.waypoints = waypoints
        self.current_speed = initial_speed
        self.acceptance_radius = acceptance_radius
        self.current_idx = 0
        self.loop = True
        self.max_accel = 1.0  # Max acceleration/deceleration rate

    def compute_velocity_commands(self, current_state, dt: float) -> Tuple[float, float]:
        if self.current_idx >= len(self.waypoints):
            return 0.0, 0.0

        target_x, target_y, target_speed = self.waypoints[self.current_idx]
        target = np.array([target_x, target_y])
        pos = np.array([current_state.x, current_state.y])
        dist = np.linalg.norm(target - pos)

        # Check if reached
        if dist < self.acceptance_radius:
            self.current_idx += 1
            if self.loop and self.current_idx >= len(self.waypoints):
                self.current_idx = 0
            return self.compute_velocity_commands(current_state, dt)

        # LOS Guidance for sharp turns
        desired_heading = np.arctan2(target[1] - pos[1], target[0] - pos[0])
        heading_error = ssa(desired_heading - current_state.yaw)
        
        # Aggressive P-Controller for tight maneuvers
        target_yaw_rate = 2.0 * heading_error 

        # Speed limit control (acceleration/deceleration)
        speed_diff = target_speed - self.current_speed
        accel = np.clip(speed_diff / dt, -self.max_accel, self.max_accel)
        self.current_speed += accel * dt

        return self.current_speed, target_yaw_rate
