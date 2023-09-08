import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Arrow
from pyorca import Agent, orca
import time
import argparse

# Create a list of agents (robots)
agents = [
    Agent(position=[3.0, 0.0], velocity=[-1.0, 0.0], radius=0.5, max_speed=1.0, pref_velocity=[-0.7, 0.0]),
    Agent(position=[0.0, 0.0], velocity=[1.0, 0.0], radius=0.5, max_speed=1.0, pref_velocity=[0.7, 0.0]),
    Agent(position=[2, 2], velocity=[-0.5, -0.5], radius=0.5, max_speed=1.0, pref_velocity=[0.5, -0.2]),
    # Agent(position=[1, 1], velocity=[-0.4, 0.4], radius=0.5, max_speed=1.0, pref_velocity=[-0.3, 0.4]),
    # Agent(position=[4, 4], velocity=[0.6, 0.6], radius=0.5, max_speed=1.0, pref_velocity=[0.5, 0.5]),
    # Agent(position=[0.0, 0.0], velocity=[1.0, 0.0], radius=0.5, max_speed=1.0, pref_velocity=[1.0, 0.0]),
    # Agent(position=[2.0, 0.0], velocity=[-1.0, 0.0], radius=0.5, max_speed=1.0, pref_velocity=[-1.0, 0.0]),
    # Add more agents as needed
]

# Simulation parameters
time_step = 0.1
total_time = 10.0
agent_num = 0

# Robot parameters
wheel_base = 0.45
linear_vel, ang_vel = 0, 0

# Matplot
figure, ax = plt.subplots()
# ax.set_xlim(-10, 10)
# ax.set_ylim(-10, 10)

# Legends
def f(m, c):
    return plt.plot([], [], marker=m, color=c, ls="none")[0]
handles, labels = [], []
colors = ["lime", "lightsalmon", "mediumpurple"]

# Add text, positions, color handles, arrow
text = []
holonomic_pos, non_holonomic_pos = [], []
velocity_arrows, lines = [], []
for idx in range(len(agents)):
    holonomic_pos.append(agents[idx].position)
    non_holonomic_pos.append(agents[idx].position)
    text.append(ax.text(0, 0, '', va='center', ha='center', fontsize=15, fontweight='bold'))
    handles.append(f("o", colors[idx]))
    labels.append(f'Robot {idx}')
holonomic_pos = np.array(holonomic_pos)
non_holonomic_pos = np.array(non_holonomic_pos)

plt.legend(handles, labels, loc=0, framealpha=1)


def plot_robot_behavior(type, behavior):
    # Simulate the scenario
    global velocity_arrows
    for t in np.arange(0.1, total_time, time_step):
        for agent in agents:
            for idx in range(len(agents)):
                if np.array_equal(agent.position, agents[idx].position):
                    agent_num = idx

            # Compute ORCA solution for the agent
            colliding_agents = [other_agent for other_agent in agents if other_agent != agent]
            new_velocity, half_planes = orca(agent, colliding_agents, t, time_step, behavior)

            # Update agent's velocity
            agent.velocity = new_velocity

            # Update agent's position using np.add
            np.add(agent.position, agent.velocity * time_step, out=agent.position, casting="unsafe")

            vel_x, vel_y = new_velocity[0], new_velocity[1]
            ang_vel = vel_x / wheel_base

            # Update agent positions
            np.add(holonomic_pos[agent_num], agent.velocity * time_step, out=holonomic_pos[agent_num], casting="unsafe")
            non_holonomic_pos[agent_num] = np.array([non_holonomic_pos[agent_num][0]+vel_x*np.cos(ang_vel*time_step),
                                                     non_holonomic_pos[agent_num][1]+vel_x*np.sin(ang_vel*time_step)])

            # Plot agent positions
            if type == 'position':
                if behavior == 'holonomic':
                    ax.plot(holonomic_pos[agent_num][0], holonomic_pos[agent_num][1], 'o', markersize=8,
                            label=f'Non holonomic Path {agent_num}',
                            color=colors[agent_num], alpha=0.3)
                    text[agent_num].set_position((holonomic_pos[agent_num][0], holonomic_pos[agent_num][1]))
                    text[agent_num].set_text(f'{agent_num}')
                else:
                    ax.plot(non_holonomic_pos[agent_num][0], non_holonomic_pos[agent_num][1], 'o', markersize=8,
                            label=f'Non holonomic Path {agent_num}',
                            color=colors[agent_num], alpha=0.3)
                    text[agent_num].set_position((non_holonomic_pos[agent_num][0], non_holonomic_pos[agent_num][1]))
                    text[agent_num].set_text(f'{agent_num}')

                ax.set_xlabel('Position X')
                ax.set_ylabel('Position Y')

            # Plot the half-planes and velocities
            else:
                for idx, line in enumerate(half_planes):
                    print(line)
                    # Plot the half-plane as a line
                    # ax.plot([line.point[1], line.point[1] + line.direction[1]],
                    #         [line.point[0], line.point[0] + line.direction[0]], label=f'Half-Plane {agent_num}')
                    l = ax.plot([line.point[0], line.point[0] + line.direction[0]],
                            [line.point[1], line.point[1] + line.direction[1]], label=f'Half-Plane {agent_num}',
                            color=colors[agent_num])
                    # text[agent_num].set_position(((line.point[0] + (line.point[0] + line.direction[0]))/2, (line.point[1] + (line.point[1] + line.direction[1]))/2))
                    # text[agent_num].set_text(f'{agent_num}')
                    lines.append(l)

                # Plot the avoidance velocities
                velocity_arrow = ax.arrow(agent.velocity[0], agent.velocity[1], 0.1, 0.1,
                         head_width=0.1, head_length=0.1, fc=colors[agent_num], ec=colors[agent_num])
                text[agent_num].set_position((agent.velocity[0], agent.velocity[1]))
                text[agent_num].set_text(f'{round(np.linalg.norm(agent.velocity), 3)}')

                velocity_arrows.append(velocity_arrow)

                ax.set_xlabel('Velocity X')
                ax.set_ylabel('Velocity Y')

        ax.set_title(f'Time: {t:.1f} seconds')
        ax.grid(True)
        plt.pause(0.1)

        for line in lines:
            for l in line:
                l.remove()

        for arrow in velocity_arrows:
            arrow.remove()

        lines.clear()
        velocity_arrows.clear()

    plt.axis('equal')
    plt.show()  # Display the final plot


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ORCA Simulation')

    parser.add_argument('--plot_type', type=str, default='position', help='Plot positions or velocity')
    parser.add_argument('--behavior', type=str, default='holonomic', help='Robot behavior')

    args = parser.parse_args()

    # Plot type: position, velocity
    # Behavior: holonomic, non_holonomic
    plot_robot_behavior(args.plot_type, args.behavior)



