import gym
import math
import networkx as nx
import heapq

### submission information ####
TEAM_NAME = "GRENECHE Lucas"
##############################

##############################
# Global Variables 
paths = {} # Dict, key = Agent_id, value = A* path
last_episode = -1
path_idx = {} # Dict, key = Agent_id, value = current node index in paths
last_node = {} # To detect when an agent has reached its next waypoint

#STUCK_THREESHOLD = 1
WAIT_COST = 5  # ex: 10
#constraints = set() # Every constraints for a Node, key = Node, value = Constraint

## A_STAR PATHFINDING

class CT_Node:
    """Class for the nodes in constraints tree"""
    def __init__(self, constraints, solution, cost):
        self.constraints = constraints 
        self.solution = solution
        self.cost = cost


## UTILITY FUNCTIONS
def h_euclidian(pos, u, v):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return (math.sqrt((x1 - x2) ** 2 + (y1 - y2)**2))



def compute_sic(env, solution):
    sic = 0
    for agent in solution:
        path = solution[agent]
        for t in range(len(path) - 1):
            if path[t] != path[t+1]:
                sic += env.G[path[t]][path[t+1]]['weight']
            else:
                sic += WAIT_COST
    return sic

# Utility functions for detect_conflict
def get_node_at(solution, agent_id, t):
    if agent_id not in solution:
        return None
    path = solution[agent_id]
    if t >= len(path):
        return path[-1]
    return path[t]


def a_star_constrained(env, agent, start, goal, constraints):
    """A* pathfinding in env.G, if node in constraints, avoided"""
    ## Inititalisation
    h = h_euclidian(env.pos,start,goal)
    counter = 0
    open_list = [(h,counter,start,0,[start],0)] # (f,counter,noeud,t,chemin,g)
    visited = set()
    constraint_set = constraints
    max_t = len(list(env.G.nodes())) * env.agent_num

    # Latest timestep at which the goal node is constrained for this agent
    # The agent must arrive AFTER this time
    goal_constraints_times = [c[2] for c in constraint_set if c[0] == agent and c[1] == goal]
    min_goal_arrival = max(goal_constraints_times) + 1 if goal_constraints_times else 0

    # Principal Loop
    while open_list != []:
        (f,_,node,t,path,g) = heapq.heappop(open_list)
        if node == goal and t >= min_goal_arrival:
            return path
        if ((node,t) in visited):
                continue
        visited.add((node,t))

        # Neighbor Expansion
        for neighbor in env.G.neighbors(node):
            # We check if the move is constrained for the agent at time t+1, if not we add it to the open list
            # g value is the cost from start to neighbor, f value is g + h (heuristic)
            if (agent,neighbor,t+1) not in constraint_set and t+1 <= max_t:
                counter += 1
                new_g = g + env.G[node][neighbor]['weight']
                new_f = new_g + h_euclidian(env.pos, neighbor,goal)
                heapq.heappush(open_list, (new_f, counter, neighbor, t+1, path+[neighbor], new_g))

        # We also consider the possibility of waiting on the current node, if it's not constrained
        if (agent,node,t+1) not in constraint_set and t+1 <= max_t:
            counter += 1
            new_g = g
            new_f = new_g + h_euclidian(env.pos, node,goal)
            heapq.heappush(open_list, (new_f, counter, node, t+1, path+[node], new_g))

    # Fallback: unconstrained A* so agent still has a valid path (CBS will detect conflicts and continue)
    if start == goal:
        return [start]
    try:
        return nx.astar_path(
            env.G, start, goal,
            heuristic=lambda u, v: h_euclidian(env.pos, u, v),
            weight='weight'
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return [start]

def apply_safety(env, actions):
    """From MASAHIRO KAJI's SafeDRP work.
    """
    do = True
    while do:
        do = False
        for i in range(env.agent_num):
            start_i = env.current_start[i]
            goal_i  = env.current_goal[i]

            if goal_i is None:
                # Case 1: two agents heading to the same node
                for j in range(env.agent_num):
                        if j != i and actions[i] == actions[j]:
                            if actions[i] != start_i:  # only trigger if action actually changes
                                actions[i] = start_i
                                do = True
                            break

            if goal_i is None:
                # Case 2: head-on swap
                for j in range(env.agent_num):
                        if j != i and (
                            actions[j] == env.current_start[i]
                            and actions[i] == env.current_start[j]
                        ):
                            if actions[i] != start_i:  # only trigger if action actually changes
                                actions[i] = start_i
                                do = True
                            break
    return actions



def detect_conflict(solution):
    """Detect if there is a conflict in the solution, return the first concflict found as a tuple (agent1, agent2, node, time_step)"""
    agents = list(solution.keys())
    # We check up to a reasonable time horizon (longest path + buffer for waiting agents)
    max_t = max(len(path) for path in solution.values()) + len(agents)
    
    for t in range(max_t):
        for i_idx, i in enumerate(agents):
            for j in agents[i_idx+1:]:
                node_i = get_node_at(solution, i, t)
                node_j = get_node_at(solution, j, t)
                
                # Vertex conflict
                if node_i == node_j:
                    return ('vertex', i, j, node_i, t)
        
                # Edge conflict (swap)
                if t < max_t - 1:
                    next_i = get_node_at(solution, i, t+1)
                    next_j = get_node_at(solution, j, t+1)
                    if node_i == next_j and node_j == next_i:
                        return ('edge', i, j, node_i, t)
    
    return None




def make_root_node(env):
    """Root_Node creation"""

    # calculation of a_star path for every agent
    for agent in range(env.agent_num):
        current = env.current_start[agent]
        end = env.goal_array[agent]

        paths[agent] = a_star_constrained(env,agent,current,end,set())

    solution = paths.copy()
    sic = compute_sic(env,solution)

    root = CT_Node(set(),solution,sic)

    return root
    

def cbs(env):
    ## Creation of the root node
    root = make_root_node(env)
    open_list = []
    counter = 0
    max_iter = 5000 # Increased for better results
    iter_count = 0

    # Push the root node in the queue
    heapq.heappush(open_list, (root.cost, counter, root))

    while open_list and iter_count < max_iter:
        iter_count += 1
        (_,__,ct_node) = heapq.heappop(open_list)
        solution = ct_node.solution
        conflict = detect_conflict(solution)
        if conflict is None:
            return solution

        conflict_type = conflict[0]
        agent1 = conflict[1]
        agent2 = conflict[2]

        # Contraintes selon le type de conflit
        if conflict_type == 'vertex':
            node = conflict[3]
            time = conflict[4]
            c1 = (agent1, node, time)
            c2 = (agent2, node, time)
        else:  # edge conflict : agent1 va vers node_j, agent2 va vers node_i
            node_i = conflict[3]  # position de agent1 au temps t = destination de agent2
            node_j = get_node_at(solution, agent2, conflict[4])  # position de agent2 au temps t
            time = conflict[4]
            c1 = (agent1, node_j, time + 1)  # agent1 interdit à node_j au temps t+1
            c2 = (agent2, node_i, time + 1)  # agent2 interdit à node_i au temps t+1

        ## CHILD1
        child1_constraints = ct_node.constraints | {c1}
        new_path1 = a_star_constrained(env, agent1, env.current_start[agent1], env.goal_array[agent1], child1_constraints)
        solution1 = dict(ct_node.solution)
        solution1[agent1] = new_path1
        sic1 = compute_sic(env, solution1)
        child1 = CT_Node(child1_constraints, solution1, sic1)
        counter += 1
        heapq.heappush(open_list, (child1.cost, counter, child1))

        ## CHILD2
        child2_constraints = ct_node.constraints | {c2}
        new_path2 = a_star_constrained(env, agent2, env.current_start[agent2], env.goal_array[agent2], child2_constraints)
        solution2 = dict(ct_node.solution)
        solution2[agent2] = new_path2
        sic2 = compute_sic(env, solution2)
        child2 = CT_Node(child2_constraints, solution2, sic2)
        counter += 1
        heapq.heappush(open_list, (child2.cost, counter, child2))

    return None

def init(env):
    global last_episode, paths, path_idx, last_node
    last_episode = env.episode_account # Initialization of the episode number
    paths.clear()
    path_idx.clear()
    last_node.clear()

    ## CBS
    result = cbs(env)
    if result is not None:
        paths = result
    else:
        # Fallback: Standard Priority-based A* (more robust than simple A*)
        blocked = set()
        for agent in range(env.agent_num):
            p = a_star_constrained(env, agent, env.current_start[agent], env.goal_array[agent], 
                                   {(agent, b, t) for b in blocked for t in range(100)})
            paths[agent] = p
            blocked.add(env.goal_array[agent])
    
    for agent in range(env.agent_num):
        path_idx[agent] = 0
        last_node[agent] = env.current_start[agent]

def policy(obs, env):
    global last_episode

    if (env.episode_account != last_episode):
        init(env)

    actions = []
    for agent in range(env.agent_num):
        if env.current_goal[agent] is not None:
            actions.append(env.current_goal[agent])
        else:
            # Agent is at a node, check if we need to advance the path index
            curr = env.current_start[agent]
            path = paths[agent]
            
            if path_idx[agent] < len(path) - 1 and curr == path[path_idx[agent]]:
                 path_idx[agent] += 1
            
            actions.append(path[path_idx[agent]])

    actions = apply_safety(env, actions)

    return actions
