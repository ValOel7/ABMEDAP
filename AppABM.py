import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import networkx as nx

# Initialize simulation parameters
def get_model_params():
    return {
        "N": st.sidebar.slider("Number of agents", 50, 500, 100),
        "initial_infected": st.sidebar.slider("Initial Number of Affected Assets", 1, 10, 3),
        "infection_probability": st.sidebar.slider("Infection Probability", 0.0, 1.0, 0.5),
        "steps": st.sidebar.slider("Experiment Duration (Seconds)", 5, 100, 50),  # Duration of the experiment
    }

# Simple Moving Average function for smoothing
def moving_average(data, window_size=1):
    if len(data) < window_size:
        return data
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Agent class
class Agent:
    def __init__(self, unique_id, status, size):
        self.unique_id = unique_id
        self.status = status  # "affected assets", "susceptible", "asset back equilibrium", "asset liquidated"
        self.size = size  # Determines susceptibility
        self.infection_timer = 0  # Timer for conversion delay
        self.recovery_timer = 0  # Timer for turning green after infection

    def interact(self, neighbors, infection_probability):
        if self.status == "affected assets":
            for neighbor in neighbors:
                if neighbor.status == "susceptible":
                    susceptibility_factor = 1.0 / neighbor.size  # Smaller nodes are more susceptible
                    if random.random() < (infection_probability * susceptibility_factor):
                        neighbor.infection_timer = self.size  # Delay based on size

    def update_status(self):
        if self.status == "susceptible" and self.infection_timer > 0:
            self.infection_timer -= 1
            if self.infection_timer == 0:
                self.status = "affected assets"
                self.recovery_timer = 3  # Stay affected for 3 seconds before recovering
        elif self.status == "affected assets" and self.recovery_timer > 0:
            self.recovery_timer -= 1
            if self.recovery_timer == 0:
                self.status = "asset back equilibrium" if random.random() > 0.5 else "asset liquidated"  # 50% chance to turn into asset back equilibrium or liquidated

# Disease Spread Model
class DiseaseSpreadModel:
    def __init__(self, **params):
        self.num_agents = params["N"]
        self.infection_probability = params["infection_probability"]
        self.G = nx.barabasi_albert_graph(self.num_agents, 3)
        self.agents = {}
        
        all_nodes = list(self.G.nodes())
        initial_infected = random.sample(all_nodes, params["initial_infected"])  # Select initial affected nodes
        
        for node in all_nodes:
            size = random.choice([1, 2, 3, 4])  # 1 (most susceptible) to 4 (least susceptible)
            status = "affected assets" if node in initial_infected else "susceptible"
            self.agents[node] = Agent(node, status, size)
        
        self.node_positions = nx.spring_layout(self.G)  # Fix network shape
        self.history = []
        self.infection_counts = []
        self.equilibrium_counts = []
        self.liquidated_counts = []

    def step(self, step_num):
        infections = 0
        newly_equilibrium = 0
        newly_liquidated = 0
        
        for node, agent in self.agents.items():
            neighbors = [self.agents[n] for n in self.G.neighbors(node)]
            agent.interact(neighbors, self.infection_probability)
        
        for agent in self.agents.values():
            prev_status = agent.status
            agent.update_status()
            if prev_status == "susceptible" and agent.status == "affected assets":
                infections += 1
            elif prev_status == "affected assets" and agent.status == "asset back equilibrium":
                newly_equilibrium += 1
            elif prev_status == "affected assets" and agent.status == "asset liquidated":
                newly_liquidated += 1
        
        self.infection_counts.append(infections)
        self.equilibrium_counts.append(newly_equilibrium)
        self.liquidated_counts.append(newly_liquidated)
        self.history.append({node: agent.status for node, agent in self.agents.items()})

# Visualization function
def plot_visuals(G, agents, positions, infections, equilibrium_counts, liquidated_counts):
    color_map = {"affected assets": "red", "susceptible": "gray", "asset back equilibrium": "green", "asset liquidated": "blue"}
    node_colors = [color_map[agents[node].status] for node in G.nodes()]
    node_sizes = [agents[node].size * 50 for node in G.nodes()]  # Adjust node size by susceptibility
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Network plot
    nx.draw(G, pos=positions, ax=axes[0, 0], node_color=node_colors, with_labels=False, node_size=node_sizes, edge_color="gray")
    axes[0, 0].set_title("Asset Spread Network")
    
    # Infection time series plot
    axes[0, 1].plot(moving_average(infections), color="red", linewidth=1.5)
    axes[0, 1].set_title("Affected Assets Over Time")
    axes[0, 1].set_xlabel("Time (Seconds)")
    axes[0, 1].set_ylabel("New Affected Assets per Step")
    
    # Asset back equilibrium time series plot
    axes[1, 0].plot(moving_average(equilibrium_counts), color="green", linewidth=1.5)
    axes[1, 0].set_title("New Asset Back Equilibrium Per Step")
    axes[1, 0].set_xlabel("Time (Seconds)")
    axes[1, 0].set_ylabel("Equilibrium Count Per Step")
    
    # Asset liquidated time series plot
    axes[1, 1].plot(moving_average(liquidated_counts), color="blue", linewidth=1.5)
    axes[1, 1].set_title("New Asset Liquidated Per Step")
    axes[1, 1].set_xlabel("Time (Seconds)")
    axes[1, 1].set_ylabel("Liquidated Count Per Step")
    
    plt.tight_layout()
    return fig

# Streamlit App
st.title("Scale-Free Network Asset Spread Simulation")
params = get_model_params()

if st.button("Run Simulation"):
    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
    model = DiseaseSpreadModel(**params)
    progress_bar = st.progress(0)
    visual_plot = st.empty()
    
    for step_num in range(1, params["steps"] + 1):
        model.step(step_num)
        progress_bar.progress(step_num / params["steps"])
        fig = plot_visuals(model.G, model.agents, model.node_positions, model.infection_counts, model.equilibrium_counts, model.liquidated_counts)
        visual_plot.pyplot(fig)
    
    st.write("Simulation Complete.")
