import numpy as np
import math
import matplotlib.pyplot as plt
from pyorca import Agent, orca
import argparse
from halfplaneintersect import halfplane_optimize, Line
from matplotlib.patches import Polygon

# Create a list of agents (robots)

# Define constants
num_agents = 5
circle_radius = 5.0  # Radius of the circular formation
agent_radius = 0.5
max_speed = 1.0

# Calculate the angular spacing between agents
angular_spacing = 2 * math.pi / num_agents

# Create agents in a circular formation with goals opposite to their positions
agents = []
for i in range(num_agents):
    # Calculate the initial position on the circle
    angle = i * angular_spacing
    x = circle_radius * math.cos(angle)
    y = circle_radius * math.sin(angle)
    # Calculate the goal position (opposite direction)
    goal_x = -x
    goal_y = -y
    # Calculate the initial velocity (tangent to the circle)
    initial_velocity = [-y, x]  # Rotate by 90 degrees
    agent = Agent(
        position=[x, y],
        velocity=initial_velocity,
        radius=agent_radius,
        max_speed=max_speed,
        pref_velocity=initial_velocity,  # Initial preferred velocity is along the circle
        goal_pos=[goal_x, goal_y]
    )
    agents.append(agent)

# Simulation parameters
time_step = 0.1
total_time = 9.0
agent_num = 0

# Robot parameters
wheel_base = 0.45
linear_vel, ang_vel = 0, 0
MAX_ANGULAR_SPEED = 0.5

# Matplot
figure, ax = plt.subplots()

# Legends
def f(m, c):
    return plt.plot([], [], marker=m, color=c, ls="none")[0]
handles, labels = [], []
colors = ["lime", "lightsalmon", "mediumpurple", "orange", "green"]

# Add text, positions, color handles, arrow
text = []
non_holonomic_pos = []
velocity_arrows, lines, vel_area = [], [], []
for idx in range(len(agents)):
    non_holonomic_pos.append(agents[idx].position)
    text.append(ax.text(0, 0, '', va='center', ha='center', fontsize=15, fontweight='bold'))
    handles.append(f("o", colors[idx]))
    labels.append(f'Robot {idx}')
non_holonomic_pos = np.array(non_holonomic_pos)
plt.legend(handles, labels, loc=0, framealpha=1)
all_pos = np.array([0,0,0])

def collision_avoidance_control(agent, goal_pos, agents):
    # Parameters for ORCA
    neighbor_dist = 2.0  # Maximum distance to consider an agent as a neighbor
    time_horizon = 5.0  # Time horizon for collision avoidance
    max_speed = agent.max_speed

    # Find neighboring agents within the specified distance
    neighbors = [other_agent for other_agent in agents if agent != other_agent and
                 np.linalg.norm(np.array(agent.position) - np.array(other_agent.position)) < neighbor_dist]

    # Initialize lists for ORCA constraints
    half_planes = []

    # Add ORCA constraints for each neighboring agent
    for neighbor in neighbors:
        relative_position = np.array(agent.position) - np.array(neighbor.position)
        relative_velocity = np.array(agent.velocity) - np.array(neighbor.velocity)
        combined_radius = agent.radius + neighbor.radius

        # Calculate the distance between the agents' centers
        dist = np.linalg.norm(relative_position)
        if dist < combined_radius:
            # Agents are too close; they must avoid collision
            continue  # Skip this agent to avoid dividing by zero (handle this case appropriately)

        # Compute the relative velocity in the direction of relative_position
        relative_speed = np.dot(relative_velocity, relative_position) / dist

        # Check if the agents are moving towards each other
        if relative_speed < 0:
            # ORCA constraint
            t_c = dist / relative_speed  # Time to collision
            if t_c < time_horizon:
                # Calculate the point of collision avoidance
                point = agent.position + (relative_position / dist) * (0.5 * (t_c - 1.0))
                v_rel = np.array(agent.velocity) - np.array(neighbor.velocity)
                line = Line(point, v_rel)
                half_planes.append(line)

    # Use ORCA to compute a new velocity for the agent
    if len(half_planes) > 0:
        new_velocity = halfplane_optimize(half_planes, agent.velocity)
    else:
        # No collision risk, move directly towards the goal
        error_x = goal_pos[0] - agent.position[0]
        error_y = goal_pos[1] - agent.position[1]
        angle_to_goal = np.arctan2(error_y, error_x)
        heading_error = angle_to_goal - agent.velocity[1]
        kp_angular = 0.1
        angular_velocity = kp_angular * heading_error
        new_velocity = [max_speed, angular_velocity]
    return new_velocity

def calculate_non_holonomic_control(agent, goal_pos, collision_risk):
    # Calculate errors
    error_x = goal_pos[0] - agent.position[0]
    error_y = goal_pos[1] - agent.position[1]

    # Calculate distance and angle to the goal
    distance_to_goal = np.sqrt(error_x**2 + error_y**2)
    distance_threshold = 1.0
    angle_to_goal = np.arctan2(error_y, error_x)

    if distance_to_goal < distance_threshold:
        # Agent is very close to the goal, set velocity to zero
        linear_velocity, angular_velocity = 0.0, 0.0
    else:
        # Check if there is a collision risk (e.g., using ORCA)
        if collision_risk:
            # Apply collision avoidance logic (e.g., ORCA)
            linear_velocity, angular_velocity = collision_avoidance_control(agent, goal_pos, agents)
        else:
            # Calculate linear velocity based on the magnitude of the velocity vector
            linear_velocity = np.linalg.norm(agent.velocity)

            # Calculate the angle difference to steer towards the goal
            angle_difference = angle_to_goal - agent.velocity[1]

            # Ensure the angular velocity is positive to steer in the correct direction
            kp_angular = 0.1
            angular_velocity = kp_angular * angle_difference

            # Limit the angular velocity to a maximum value
            max_angular_velocity = 0.5  # You can adjust this value as needed
            angular_velocity = np.clip(angular_velocity, -max_angular_velocity, max_angular_velocity)

            # Determine the direction of motion (forward or backward)
            if np.cos(angle_difference) < 0:
                linear_velocity *= -1  # Reverse direction if the angle is greater than 90 degrees

    return linear_velocity, angular_velocity

def calculate_new_position(agent, time_step):
    global all_pos
    # Update agent's position based on linear and angular velocities
    linear_velocity, angular_velocity = agent.velocity

    # Calculate the new position based on velocities and time_step
    new_x = agent.position[0] + linear_velocity * np.cos(all_pos[2]) * time_step
    new_y = agent.position[1] + linear_velocity * np.sin(all_pos[2]) * time_step
    new_theta = all_pos[2] + angular_velocity * time_step

    return np.array([new_x, new_y, new_theta])

def plot_robot_behavior(type):
    # Simulate the scenario
    global velocity_arrows, all_pos
    for t in np.arange(0.1, total_time, time_step):
        for agent in agents:
            agent_num = agents.index(agent)
            print(f"Agent {agent_num} - Position: {agent.position}, Velocity: {agent.velocity} at time: {t} and with goal pos: {agent.goal_pos}")

            # Compute ORCA solution for the agent
            colliding_agents = [other_agent for other_agent in agents if other_agent != agent]
            new_velocity, half_planes, collision_risk = orca(agent, colliding_agents, t, time_step)

            # Update agent's velocity and position for goal-seeking behavior
            if args.behavior == 'non_holonomic':
                linear_velocity, angular_velocity = calculate_non_holonomic_control(agent, agent.goal_pos, collision_risk)
                agent.velocity = np.array([linear_velocity, angular_velocity])
                new_position = calculate_new_position(agent, time_step)
                agent.position = new_position[:2]
                all_pos = new_position

            # Plot agent positions
            if type == 'position':
                ax.plot(agent.position[0], agent.position[1], 'o', markersize=8,
                        label=f'Non-Holonomic Path {agent_num}',
                        color=colors[agent_num], alpha=0.3)
                text[agent_num].set_position((agent.position[0], agent.position[1]))
                text[agent_num].set_text(f'{agent_num}')

                ax.set_xlabel('Position X')
                ax.set_ylabel('Position Y')

            # Plot the half-planes and velocities
            else:
                for idx, line in enumerate(half_planes):
                    l = ax.plot([line.point[0], line.point[0] + line.direction[0]],
                                [line.point[1], line.point[1] + line.direction[1]], label=f'Half-Plane {agent_num}',
                                color=colors[agent_num])
                    lines.append(l)

                    # Plot the permissible velocity area as a polygon
                    max_speed = agent.max_speed
                    max_angular_speed = MAX_ANGULAR_SPEED
                    center = agent.position
                    radius = max_speed * time_step
                    angular_range = np.linspace(-max_angular_speed * time_step, max_angular_speed * time_step, 100)
                    velocity_area = np.array([[
                        center[0] + radius * np.cos(angle),
                        center[1] + radius * np.sin(angle)
                    ] for angle in angular_range])

                    p = ax.plot(velocity_area[:, 0], velocity_area[:, 1], linestyle='--', color=colors[agent_num], alpha=1.0)
                    vel_area.append(p)

                # Plot the avoidance velocities
                # velocity_arrow = ax.arrow(agent.velocity[0], agent.velocity[1], 0.1, 0.1,
                #          head_width=0.1, head_length=0.1, fc=colors[agent_num], ec=colors[agent_num])
                # text[agent_num].set_position((agent.velocity[0], agent.velocity[1]))
                # text[agent_num].set_text(f'{round(np.linalg.norm(agent.velocity), 3)}')
                # velocity_arrows.append(velocity_arrow)

                ax.set_xlabel('Velocity X')
                ax.set_ylabel('Velocity Y')

        ax.set_title(f'Time: {t:.1f} seconds')
        ax.grid(True)
        plt.pause(0.1)

        for line in lines:
            for l in line:
                l.remove()
        for pol in vel_area:
            for p in pol:
                p.remove()
        # for arrow in velocity_arrows:
        #     arrow.remove()

        lines.clear()
        vel_area.clear()
        # velocity_arrows.clear()

    plt.axis('equal')
    plt.show()  # Display the final plot

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA Simulation')
    parser.add_argument('--plot_type', type=str, default='position', help='Plot positions or velocity')
    parser.add_argument('--behavior', type=str, default='non_holonomic', help='Robot behavior')
    args = parser.parse_args()
    # Plot type: position, velocity
    # Behavior: non_holonomic
    plot_robot_behavior(args.plot_type)