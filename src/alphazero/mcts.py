"""MCTS implementation for AlphaZero."""
import math
import numpy as np
import torch
from typing import Dict, List, Tuple, Optional

from src.dqn.actions import NUM_ACTIONS, get_action_mask
from src.alphazero.net import AlphaZeroNet

class Node:
    def __init__(self, prior: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior = prior
        self.children: Dict[int, Node] = {}
        
    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count

class MCTS:
    def __init__(self, network: AlphaZeroNet, device: str, c_puct: float = 1.0):
        self.network = network
        self.device = device
        self.c_puct = c_puct
        
    def run(self, root_state: Dict, num_simulations: int, env_simulator) -> np.ndarray:
        """
        Runs MCTS simulations and returns visit counts (policy).
        
        Args:
            root_state: Dict with 'embeddings', 'mask', 'mistakes_left'
            num_simulations: Number of simulations to run
            env_simulator: Object/function to simulate environment steps (copy of env)
        """
        root = Node(0)
        
        # Expand root immediately
        self._expand(root, root_state, env_simulator)
        
        for _ in range(num_simulations):
            node = root
            sim_env = env_simulator.copy() # Need a way to copy env state cheaply
            
            search_path = [node]
            
            # Selection
            while node.children:
                action, node = self._select_child(node)
                search_path.append(node)
                # Step simulation
                # Note: This requires the simulator to be stateful or we track state
                # For Connections, state is deterministic given action history?
                # Actually, we just need to update mask and mistakes.
                # Let's assume sim_env has a step() that updates its internal state
                sim_env.step(action)
                
            # Expansion & Evaluation
            # Get leaf state from simulator
            leaf_state = sim_env.get_state() 
            
            # Check if terminal
            if sim_env.is_done():
                value = sim_env.get_result_value() # +1 or -1
            else:
                value = self._expand(node, leaf_state, sim_env)
                
            # Backpropagation
            self._backpropagate(search_path, value)
            
        # Compute policy from root visit counts
        policy = np.zeros(NUM_ACTIONS)
        for action, child in root.children.items():
            policy[action] = child.visit_count
            
        policy /= np.sum(policy)
        return policy

    def _select_child(self, node: Node) -> Tuple[int, Node]:
        """Selects the child with the highest UCB score."""
        best_score = -float('inf')
        best_action = -1
        best_child = None
        
        for action, child in node.children.items():
            ucb = self._ucb_score(node, child)
            if ucb > best_score:
                best_score = ucb
                best_action = action
                best_child = child
                
        return best_action, best_child
        
    def _ucb_score(self, parent: Node, child: Node) -> float:
        pb_c = math.log((parent.visit_count + 19652 + 1) / 19652) + 1.25
        pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
        
        prior_score = child.prior * pb_c
        value_score = child.value()
        
        return prior_score + value_score

    def _expand(self, node: Node, state: Dict, env_simulator) -> float:
        """Expands the node using the network."""
        # Prepare input
        embeddings = torch.FloatTensor(state["embeddings"]).unsqueeze(0).to(self.device)
        mask = torch.FloatTensor(state["mask"]).unsqueeze(0).to(self.device)
        mistakes = torch.FloatTensor([state["mistakes_left"]]).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            logits, value = self.network(embeddings, mask, mistakes)
            
        # Mask invalid actions
        action_mask = env_simulator.get_action_mask()
        torch_mask = torch.BoolTensor(action_mask).to(self.device)
        logits[0, ~torch_mask] = float("-inf")
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        
        # Create children for valid actions
        valid_indices = np.where(action_mask)[0]
        for action in valid_indices:
            node.children[action] = Node(probs[action])
            
        return value.item()

    def _backpropagate(self, search_path: List[Node], value: float):
        """Backpropagates the value up the search path."""
        for node in search_path:
            node.value_sum += value
            node.visit_count += 1

# Helper for simulation
class EnvSimulator:
    """Lightweight simulator for MCTS."""
    def __init__(self, embeddings, mask, mistakes_left, puzzle_data):
        self.embeddings = embeddings
        self.mask = mask.copy()
        self.mistakes_left = mistakes_left
        self.puzzle_data = puzzle_data # Need words/groups to check correctness
        self.done = False
        self.result = 0.0
        
        # Build group sets (copied from env.py)
        self.group_sets = [set() for _ in range(4)]
        for idx, g_id in enumerate(puzzle_data["group_ids"]):
            self.group_sets[g_id].add(idx)
            
    def copy(self):
        return EnvSimulator(self.embeddings, self.mask, self.mistakes_left, self.puzzle_data)
        
    def get_state(self):
        return {
            "embeddings": self.embeddings,
            "mask": self.mask.copy(),
            "mistakes_left": self.mistakes_left
        }
        
    def get_action_mask(self):
        return get_action_mask(self.mask)
        
    def is_done(self):
        return self.done or self.mistakes_left <= 0 or np.all(self.mask)
        
    def get_result_value(self):
        if np.all(self.mask):
            return 1.0
        if self.mistakes_left <= 0:
            return -1.0
        return 0.0 # Should not happen if called on terminal
        
    def step(self, action_idx):
        # Simplified step logic (no reward calculation, just state update)
        from src.dqn.actions import get_action_from_idx
        indices = get_action_from_idx(action_idx)
        guess_set = set(indices)
        
        best_overlap = 0
        for g_set in self.group_sets:
            overlap = len(guess_set & g_set)
            if overlap > best_overlap:
                best_overlap = overlap
                
        if best_overlap == 4:
            for i in indices:
                self.mask[i] = True
        else:
            self.mistakes_left -= 1
            
        if np.all(self.mask):
            self.done = True
        elif self.mistakes_left <= 0:
            self.done = True
