import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import random
import time
import networkx as nx
import random

# Initialize simulation parameters
def get_model_params():
    return {
        "N": st.sidebar.slider("Number of assets in the investment portfolio", 50, 500, 100),
        "initial_affected": st.sidebar.slider("Initial Number of Assets affected in the market downturn", 1, 10, 3),
        "affected_correlation": st.sidebar.slider("Correlation between the assets", 0.0, 1.0, 0.5),
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
        self.status = status  # "affected", "susceptible assets", "", "liquidated_asset"
        self.size = size  # Determines susceptibility
        self.infection_timer = 0  # Timer for conversion delay
        self.recovery_timer = 0  # Timer for turning green after infection

  def interact(self, neighbors, affected_correlation):
      if self.status == "affected":
         for neighbor in neighbors:
              if neighbor.status == "susceptible assets":
                 susceptibility_factor = 1.0 / neighbor.size  # Smaller nodes are more susceptible assets
                  random_correlation = affected_correlation * (random.uniform(-1, 1) / neighbor.size)
                
                  if random_correlation > 0.5:  # Adjust threshold as needed
                     neighbor.infection_timer = self.size  # Delay based on size

    def update_status(self):
        if self.status == "susceptible assets" and self.infection_timer > 0:
            self.infection_timer -= 1
            if self.infection_timer == 0:
                self.status = "affected"
                self.recovery_timer = 5  # Stay affected for 3 seconds before recovering
        elif self.status == "affected" and self.recovery_timer > 0:
            self.recovery_timer -= 1
            if self.recovery_timer == 0:
                self.status = "asset_back_equilibrium" if random.random() > 0.5 else "liquidated_asset"  # 50% chance to turn blue (liquidated_asset)

# Disease Spread Model
class RiskSpreadModel:
    def __init__(self, **params):
        self.num_agents = params["N"]
        self.random_correlation = params["random_correlation"]
        self.G = nx.barabasi_albert_graph(self.num_agents, 3)
        self.agents = {}
        
        all_nodes = list(self.G.nodes())
        initial_affected = random.sample(all_nodes, params["initial_affected"])  # Select initial affected nodes
        
        for node in all_nodes:
            size = random.choice([1, 2, 3, 4])  # 1 (most susceptible assets) to 4 (susceptible assets)
            status = "affected" if node in initial_affected else "susceptible assets"
            self.agents[node] = Agent(node, status, size)
        
        self.node_positions = nx.spring_layout(self.G)  # Fix network shape
        self.history = []
        self.affectedasset_counts = []
        self.asset_back_equilibrium_counts = []
        self.liquidated_asset_counts = []

    def step(self, step_num):
        infections = 0
        newly_asset_back_equilibrium = 0
        newly_liquidated_asset = 0
        
        for node, agent in self.agents.items():
            neighbors = [self.agents[n] for n in self.G.neighbors(node)]
            agent.interact(neighbors, self.random_correlation)
        
        for agent in self.agents.values():
            prev_status = agent.status
            agent.update_status()
            if prev_status == "susceptible assets" and agent.status == "affected":
                infections += 1
            elif prev_status == "affected" and agent.status == "asset_back_equilibrium":
                newly_asset_back_equilibrium += 1
            elif prev_status == "affected" and agent.status == "liquidated_asset":
                newly_liquidated_asset += 1
        
        self.affectedasset_counts.append(infections)
        self.asset_back_equilibrium_counts.append(newly_asset_back_equilibrium)
        self.liquidated_asset_counts.append(newly_liquidated_asset)
        self.history.append({node: agent.status for node, agent in self.agents.items()})

# Visualization function
def plot_visuals(G, agents, positions, infections, asset_back_equilibrium_counts, liquidated_asset_counts):
    color_map = {"affected": "red", "susceptible assets": "gray", "asset_back_equilibrium": "green", "liquidated_asset": "blue"}
    node_colors = [color_map[agents[node].status] for node in G.nodes()]
    node_sizes = [agents[node].size * 50 for node in G.nodes()]  # Adjust node size by susceptibility
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Network plot
    nx.draw(G, pos=positions, ax=axes[0, 0], node_color=node_colors, with_labels=False, node_size=node_sizes, edge_color="gray")
    axes[0, 0].set_title("Disease Spread Network")
    
    # Infection time series plot
    axes[0, 1].plot(moving_average(infections), color="red", linewidth=1.5)
    axes[0, 1].set_title("Infection Spread Over Time")
    axes[0, 1].set_xlabel("Time (Seconds)")
    axes[0, 1].set_ylabel("New Infections per Step")
    
    # asset_back_equilibrium time series plot
    axes[1, 0].plot(moving_average(asset_back_equilibrium_counts), color="green", linewidth=1.5)
    axes[1, 0].set_title("New asset_back_equilibrium Per Step")
    axes[1, 0].set_xlabel("Time (Seconds)")
    axes[1, 0].set_ylabel("asset_back_equilibrium Count Per Step")
    
    # liquidated_asset time series plot
    axes[1, 1].plot(moving_average(liquidated_asset_counts), color="blue", linewidth=1.5)
    axes[1, 1].set_title("New liquidated_asset Per Step")
    axes[1, 1].set_xlabel("Time (Seconds")
    axes[1, 1].set_ylabel("liquidated_asset Count Per Step")
    
    plt.tight_layout()
    return fig

# Streamlit App
st.title("Scale-Free Network Investment and Portfolio Risk Analysis Spread Simulation")
params = get_model_params()

if st.button("Run Risk Analysis"):
    st.markdown("<script>window.scrollTo(0, document.body.scrollHeight);</script>", unsafe_allow_html=True)
    model = RiskSpreadModel(**params)
    progress_bar = st.progress(0)
    visual_plot = st.empty()
    
    for step_num in range(1, params["steps"] + 1):
        model.step(step_num)
        progress_bar.progress(step_num / params["steps"])
        fig = plot_visuals(model.G, model.agents, model.node_positions, model.affectedasset_counts, model.asset_back_equilibrium_counts, model.liquidated_asset_counts)
        visual_plot.pyplot(fig)
    
    st.write("Investment and Portfolio Risk Analysis Simulation Complete.")

    st.markdown( """ Scale-Free Network Investment and Portfolio Risk Analysis Spread Simulation

This simulation models a market downturn in one sector and how an asset impacts another asset through correlation in a **scale-free network** using an agent-based approach. 
Nodes represent different assets, where **red indicates that the asset has been affetced by the market downturn, green represents the asset going back to equilibrium, and blue signifies that the asset Liquidated**. 
The market downturn follows a **proximity-based transmission**, with larger nodes taking longer to affect smaller ones. 
After 5 time steps, an affected node **goes back to equilibrium (turns green) or Liquidates (turns blue) with a 50% probability**. 
Users can adjust the number of agents, infection probability, and experiment duration.

**Visualizations:**
- A **real-time network graph** showing nodes changing color as a market turndown progresses.
- **Three smoothed time series plots**:
  - Affected assets over time (**Red**).
  - Number of assets reaching eqilibrium per step (**Green**).
  - Number of liquidations per step (**Blue**).

Adjust parameters and run the simulation to observe how a market downturn in one sector impacts correlated assets in a complex network.
"""
)
