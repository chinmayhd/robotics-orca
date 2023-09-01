import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pyorca import Agent, orca

# Create a list of agents (robots)
agents = [
    Agent(position=[-2, 2], velocity=[0.1, 0.1], radius=0.1, max_speed=1.0, pref_velocity=[-0.1, 0.1]),
    Agent(position=[0, 0], velocity=[-0.2, -0.2], radius=0.1, max_speed=1.0, pref_velocity=[0.1, 0.1]),
    # Agent(position=[2, 2], velocity=[-0.5, -0.5], radius=0.1, max_speed=1.0, pref_velocity=[0.5, 0]),
    # Add more agents as needed
]

# Create a colormap for robots
cmap = get_cmap('tab10')

# Create a single figure for the plot
plt.figure(figsize=(6, 6))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('Velocity Y')
plt.ylabel('Velocity X')
plt.grid(True)

# Turn on interactive mode
plt.ion()

# Simulation parameters
time_step = 0.1
total_time = 10.0
agent_num = 0

# Simulate the scenario
for t in np.arange(0.1, total_time, time_step):
    plt.clf()  # Clear the previous plot

    for agent in agents:
        for idx in range(len(agents)):
            if np.array_equal(agent.position, agents[idx].position):
                agent_num = idx

        # Compute ORCA solution for the agent
        colliding_agents = [other_agent for other_agent in agents if other_agent != agent]
        new_velocity, half_planes = orca(agent, colliding_agents, t, time_step)
        print(f"New velocity is {new_velocity} for agent {agent_num}")

        # Update agent's velocity
        agent.velocity = new_velocity

        # Plot the agent's velocity with color from colormap
        color = cmap(agents.index(agent))

        plt.scatter(agent.velocity[1], agent.velocity[0], label='Agent', marker='o')

        # Plot the half-planes
        for line in half_planes:
            # Plot the half-plane as a line
            plt.plot([line.point[1], line.point[1] + line.direction[1]],
                     [line.point[0], line.point[0] + line.direction[0]], label=f'Half-Plane {agent_num}')

        # Plot the avoidance velocities
        plt.arrow(agent.velocity[1], agent.velocity[0], agent.velocity[1], agent.velocity[0], head_width=0.05, head_length=0.1,
                  label='Avoidance Velocity', color=color)

    plt.title(f'Time: {t:.1f} seconds')
    plt.legend()  # Add legend
    plt.pause(0.1)  # Pause to update the plot

# Turn off interactive mode
plt.ioff()

plt.show()  # Display the final plot



