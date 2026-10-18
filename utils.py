import os
import torch
import numpy as np
from scipy.special import softmax

from model.actor import ActorOutput, Actor
from model.critic import Critic
from model.space_embed import SpaceEmbed
from model.packing_transformer import PackingTransformer

from tianshou.data import Batch
from tianshou.policy import PPOPolicy
from tianshou.utils.net.common import ActorCritic
from tianshou.policy.modelfree.ppo import PPOTrainingStats

from envs.Packing_Env.data_type import Orthogonal3D, Point3D,Cuboid

from gymnasium import spaces

from graphviz import Digraph
from copy import deepcopy
import math

from multiprocessing import Pool
import random
import matplotlib.pyplot as plt


def visualize_tree_with_graphviz(root, filename='mcts_tree'):
    """
    Visualize an MCTS tree using Graphviz.

    Args:
        root: The root node of the MCTS tree.
        filename: The output file name (without extension).
    """
    dot = Digraph(comment='MCTS Tree')

    def add_nodes_and_edges(node, parent_id=None):
        """
        Recursively add nodes and edges to the Graphviz Digraph.
        """
        node_id = str(id(node))  # Unique ID for each node
        node_label = f"State: {node.state}\nVisits: {node.visits}\nReward: {node.total_reward:.2f}"
        
        # Add the current node
        dot.node(node_id, node_label)
        
        # Add an edge from the parent to the current node (if not root)
        if parent_id is not None:
            dot.edge(parent_id, node_id)
        
        # Recursively add children
        for child in node.children:
            add_nodes_and_edges(child, node_id)
    
    # Start from the root node
    add_nodes_and_edges(root)
    
    # Render and display the tree
    # output_path = dot.render(filename, format='png', cleanup=True)
    # print(f"Tree visualization saved to {output_path}")
    # dot.view()



class CategoricalMasked(torch.distributions.Categorical):
    def __init__(self, Input:ActorOutput):
        self.device = Input.logits.device
        self.masks = Input.action_mask.type(torch.BoolTensor).to(self.device)
        self.batch_size = Input.logits.shape[0]
        logits = torch.where(
            self.masks.reshape(self.batch_size, -1), Input.logits, torch.tensor(-1e18).to(self.device)
        )
        super(CategoricalMasked, self).__init__(probs=None, logits=logits, validate_args=None)

    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks.reshape(self.batch_size, -1), p_log_p, torch.tensor(0.0).to(self.device))
        return -p_log_p.sum(-1)
    

def build_net(device="cpu"):
    net_space_embed = SpaceEmbed(embed_dim=128)
    net_pack_transform = PackingTransformer(
        embed_dim=128, ffn_expansion_factor=2, num_layers=3
    )
    actor = Actor(net_space_embed, net_pack_transform, embed_size=128, device=device)
    critic = Critic(net_space_embed, net_pack_transform, embed_size=128, device=device)
    actor_critic = ActorCritic(actor, critic)
    optim = torch.optim.Adam(actor_critic.parameters(), lr=7e-5)
    policy: PPOPolicy[PPOTrainingStats] = PPOPolicy(
        actor=actor,
        critic=critic,
        optim=optim,
        dist_fn=CategoricalMasked,
        discount_factor=1.0,
        # max_grad_norm=args.max_grad_norm,
        eps_clip=0.3,
        vf_coef=0.5,
        ent_coef=0.01,
        reward_normalization=False,
        advantage_normalization=False,
        recompute_advantage=False,
        dual_clip=None,
        value_clip=False,
        gae_lambda=0.96,
        action_space=spaces.Discrete(80),
        action_scaling=False,
        deterministic_eval=True,
    )
    curr_path = os.path.dirname(os.path.abspath(__file__))
    ck = torch.load(os.path.join(curr_path, 'checkpoint.pth'), map_location='cpu')
    policy.load_state_dict(ck['model'])
    actor = policy.actor.eval()
    critic = policy.critic.eval()
    return actor, critic


class PackingAgent:
    def __init__(self):
        self.actor, self.critic = build_net()
        self.k_placement = 80
    
    def predict(self, obs,
                value_deterministic=True, 
                logits_deterministic=True):
        data = Batch(
            new_item=obs["item"],
            ems=obs["ems"],
            action_mask=obs["mask"],
        )
        act_out, _ = self.actor(data)
        predicted_value = self.critic(data).detach().cpu().numpy().reshape(-1,)
        placable_mask = data.action_mask.sum(axis=(1,2))!=0
        if value_deterministic:
            predicted_value[placable_mask==False] = -np.inf
            item_idx = predicted_value.argmax()
        else:
            # probs = softmax(predicted_value[placable_mask])
            item_idx = np.random.choice(np.arange(len(placable_mask))[placable_mask])
        if logits_deterministic:
            act_idx = (CategoricalMasked(act_out).mode)[item_idx]
        else:
            act_idx = (CategoricalMasked(act_out).sample())[item_idx]
        
        pack_item = obs['item_raw'][item_idx]
        pos, rot = self.idx2pos(act_idx, 
                                obs['ems_unnorm'][item_idx].reshape(-1, 6))
        pack_item = pack_item[[1,0,2]] if rot else pack_item
        box = Cuboid(FLB = Point3D(*pos), Dim = Orthogonal3D(*pack_item))
        #Cuboid object, selected item and corresponding state value
        return box, (item_idx, act_idx, predicted_value[item_idx])

    def idx2pos(self, idx, candidates):
        if idx > self.k_placement-1:
            idx = idx - self.k_placement
            rot = True
        else:
            rot = False
        pos = candidates[idx][:3]
        return pos, rot

    


def parallel_rollouts(state, num_simulations=4):
    with Pool(num_simulations) as pool:
        rewards = pool.map(rollout, [state] * num_simulations)
    return sum(rewards) / num_simulations  # Average reward from rollouts        

def rollout(env, 
            agent, new_item,
            value_deterministic=True,
            logits_deterministic=True,
            w_operation = 0.01,
            w_value = 50,
            Uti_0=0.7,
            Uti_requirement=0.8):
    operations = []
    holding_list = env.container.holding_list
    items = [e.Dim.raw().tolist() for e in holding_list] + [new_item.tolist()]
    items = np.array(items)
    new_item_packed=False
    while len(items) != 0:
        pack_data = env.get_pack_data(items, offset=0.02)
        if not pack_data['placable']:
            Value = -1
            break
        else:
            box, (item_idx, act_idx,Value) = agent.predict(pack_data, 
                                                     value_deterministic=value_deterministic,
                                                     logits_deterministic=logits_deterministic)
            # print(item_idx, Value, pack_data)q
            try:
                env.pack(box)
            except Exception as e:
                print(e)
                # array_3d = env.container.State.copy()
                # fig = plt.figure()
                # ax = fig.add_subplot(111, projection='3d')
                # ax = env.render()
                # x, y, z = np.indices((10,10,10))/10*0.6+0.05
                # x = x.flatten()
                # y = y.flatten()
                # z = z.flatten()
                # values = array_3d.flatten().astype(np.int32)
                # colors = np.array(['red','green', 'blue'])
                # color_mapping = colors[values]
                # ax.scatter(x,y,z, c=color_mapping, s=50)
                # print(e)
                # env.hm.update(box=box)
                # env.render()
                # env.unpack(env.container.PlacedItems[-1])
                # env.render()
            if item_idx == len(items) - 1 and (not new_item_packed):
                holding_item_idx = None
                new_item_packed = True
            else:
                holding_item_idx = item_idx
            items = np.delete(items, item_idx, axis=0)
            env.remove_holding_item(holding_item_idx)
            operations.append([box, holding_item_idx])

    rollout_reward = env.container.Uti - Uti_0 #could be negative
    # print("rollout reward:{}".format(rollout_reward))
    
    if len(items)==0 or env.container.Uti > Uti_requirement:
    # if rollout_reward > 0 or env.container.Uti > Uti_requirement:
        Success = True
    else:
        Success = False
    # print("critic value:{}".format(Value))
    Reward = w_value * Value + rollout_reward 
    # - w_operation * len(operations)


    return Success, operations, Reward


class MCT_Node:
    def __init__(self, state, parent=None,
                 unpack_actions = [], max_child=3):
        self.state = state          # Current state of the bin packing
        self.parent = parent        # Parent node
        self.children = []          # Child nodes
        self.unexplored_actions = self.state.container.unpackableBoxes
        self.visits = 0             # Visit count
        self.total_reward = 0       # Total reward accumulated
        self.explored_actions = []
        self.sequence = []
        self.unpack_actions = unpack_actions
        self.max_child=max_child

    def compute_hash(self):
        """
        Returns a canonical (immutable) representation of the bin state
        so that rearranging cuboids in any order yields the same representation.
        """
        cuboids = self.state.container.PlacedItems
        # Sort by obj_id (or any stable unique attribute)
        sorted_cuboids = sorted(cuboids, key=lambda c: c.id)
        
        # Build a tuple of the essential attributes
        state_tuple = tuple(
            (
                cuboid.id,
                cuboid.Gx,
                cuboid.Gy,
                cuboid.Gz,
                cuboid.Gdx,
                cuboid.Gdy,
                cuboid.Gdz
            )
            for cuboid in sorted_cuboids
        )
        
        return hash(state_tuple)
    
    def is_fully_expanded(self):
        return len(self.unexplored_actions) == 0 or len(self.explored_actions) == self.max_child
    
    def perform_action(self, box):
        env_tmp = deepcopy(self.state)
        env_tmp.unpack(box)
        return env_tmp
    
    def sample_action(self):
        unpackable_box = random.sample(self.unexplored_actions, 1)[0]
        self.unexplored_actions.remove(unpackable_box)
        return unpackable_box
    
    # def sample_action(self):
    #     volumes = np.array([e.Dim.Volume for e in self.unexplored_actions])
    #     smallest_box = self.unexplored_actions[volumes.argmin()]
    #     self.unexplored_actions.remove(smallest_box)
    #     return smallest_box
    
    @property
    def num_parents(self):
        node = self
        count = 0
        while node.parent:
            count += 1
            node = node.parent
        return count
    
    def best_child(self, exploration_weight=1.0):
        # Use UCT formula to select the best child
        return max(
            self.children,
            key=lambda child: child.total_reward / (child.visits + 1e-6) + 
            exploration_weight * math.sqrt(math.log(self.visits + 1) / (child.visits + 1e-6))
        )


def mcts(initial_state, agent, new_item, iterations=30, max_unpack=10, Uti_requirement=0.8):
    root = MCT_Node(initial_state)
    transposition_table = {}
    best_sequences = []
    epoch = 0
    # for _ in range(iterations):
    while epoch < iterations:
        if len(best_sequences)>=1:
            break
        node = root
        # 1. Selection        
        while node.is_fully_expanded() and node.children:
            node = node.best_child()
        # 2. Expansion
        if node.unexplored_actions:
            if node.num_parents > max_unpack:
                node.unexplored_actions = []
                continue
            else:
                # unpackable_box = node.unexplored_actions.pop()
                # unpackable_box = random.sample(node.unexplored_actions, 1)[0]
                # node.unexplored_actions.remove(unpackable_box)
                unpackable_box = node.sample_action()
                new_state = node.perform_action(unpackable_box)
                child_node = MCT_Node(new_state, parent=node, 
                                      unpack_actions = node.unpack_actions + [unpackable_box])
                state_hash = child_node.compute_hash()
                if state_hash in transposition_table:
                    continue
                else:
                    transposition_table[state_hash] = child_node
                    node.explored_actions.append(unpackable_box)
                    node.children.append(child_node)
                    node = child_node
        
        # 3. Simulation (Rollout)
        success, operations, reward = rollout(env=deepcopy(node.state),
                                     agent=agent, new_item=new_item,
                                     Uti_0 = initial_state.container.Uti,
                                     Uti_requirement=Uti_requirement)
        if success:
            best_sequences.append((reward, operations, node.unpack_actions))
        # 4. Backpropagation
        while node:
            node.visits += 1
            node.total_reward += reward
            node = node.parent
        epoch += 1
        print('{}th iteration finished.'.format(epoch))
    # Pick the best sequence from successful ones
    # visualize_tree_with_graphviz(root)
    
    if best_sequences:
        best_sequence = max(best_sequences, key=lambda x: x[0])[1]
        unpacked_boxes = max(best_sequences, key=lambda x: x[0])[2]
        best_reward = max(best_sequences, key=lambda x: x[0])[0]
        # print('best reward:{}'.format(best_reward))
        return best_sequence, unpacked_boxes
    else:
        # print("No successful sequence found.")
        return None