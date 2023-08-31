import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from pyorca import Agent, orca

# Create a list of agents (robots)
agents = [
    Agent(position=[-4, 4], velocity=[-0.4, 0.4], radius=0.1, max_speed=1.0, pref_velocity=[1, 0]),
    Agent(position=[0, 0], velocity=[-0.4, 0.4], radius=0.1, max_speed=1.0, pref_velocity=[-1, 0]),
    Agent(position=[2, 2], velocity=[0.5, -0.5], radius=0.1, max_speed=1.0, pref_velocity=[0.5, 0]),
    # Add more agents as needed
]

# Create a colormap for robots
cmap = get_cmap('tab10')

# Create a single figure for the plot
plt.figure(figsize=(6, 6))
plt.xlim(-10, 10)
plt.ylim(-10, 10)
plt.xlabel('X')
plt.ylabel('Y')
plt.grid(True)

# Turn on interactive mode
plt.ion()

# Simulation parameters
time_step = 0.1
total_time = 10.0
cnt = 0

# Simulate the scenario
for t in np.arange(0.1, total_time, time_step):
    plt.clf()  # Clear the previous plot

    for agent in agents:
        # Compute ORCA solution for the agent
        colliding_agents = [other_agent for other_agent in agents if other_agent != agent]
        new_velocity, half_planes = orca(agent, colliding_agents, t, time_step)

        # Update agent's velocity
        agent.velocity = new_velocity

        # Update agent's position using np.add
        np.add(agent.position, agent.velocity * time_step, out=agent.position, casting="unsafe")

        # Plot the agent's position with color from colormap
        color = cmap(agents.index(agent))
        plt.plot(agent.position[0], agent.position[1], 'o', markersize=8, label=f'Robot {agents.index(agent)}',
                 color=color)

        # Plot the avoidance velocity vector as an arrow
        plt.arrow(agent.position[0], agent.position[1], agent.velocity[0], agent.velocity[1],
                  color=color, alpha=0.5, head_width=0.1, head_length=0.2)

        # Plot the half-planes with different color
        for i, line in enumerate(half_planes):
            anchor = line.point
            direction = line.direction
            plt.plot([anchor[0], anchor[0] + direction[0]],
                     [anchor[1], anchor[1] + direction[1]], '--', label=f'Half-Plane {cnt}', color=color)
            cnt += 1
            if cnt == len(agents):
                cnt = 0


    plt.title(f'Time: {t:.1f} seconds')
    plt.legend()  # Add legend
    plt.pause(0.1)  # Pause to update the plot

# Turn off interactive mode
plt.ioff()

plt.show()  # Display the final plot


