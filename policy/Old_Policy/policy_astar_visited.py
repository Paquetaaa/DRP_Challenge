import gym
import math
import networkx as nx

### submission information ####
TEAM_NAME = "GRENECHE Lucas"
##############################

##############################
# Global Variables 

#Best score : 11853 

WAIT_COST = 5
last_episode = -1
paths = {} # Dict, key = Agent_id, value = A* path
path_idx = {} # Dict, key = Agent_id, value = current position in paths[i]
last_start = {} # Dict, key = Agent_id, value = current_start
stuck = {} # Dict, key = Agent_id, value = stepspend stuck
blocked_nodes = set() # Set of nodes reached by agents as their goal

STUCK_THREESHOLD = 1
visited_nodes = {} # Dict, key = Agent_id, value = set of nodes visited by the agent
visited_nodes_twice = {} # Dict, key = Agent_id, value = set of nodes visited at least twice by the agent

## Pathfinding 
def h_euclidian(pos, u, v):
    x1, y1 = pos[u]
    x2, y2 = pos[v]
    return (math.sqrt((x1 - x2) ** 2 + (y1 - y2)**2))


def a_star(env,start,end,blocked=None):
    """A* pathfinding in env.G, if blocked is specified, nodes are avoided"""
    if start == end:
        return [start]
    G = env.G 
    if blocked:
        G = env.G.copy()
        for node in blocked:
            ## If the node is occuped, we can't put it on the path
            if node not in (start, end) and node in G.nodes():
                G.remove_node(node)
    try:
        return nx.astar_path(
            G, start, end,
            heuristic=lambda u, v: h_euclidian(env.pos, u, v),
            weight='weight'
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        if blocked:
            return a_star(env, start, end)
        return [start]
    

def init(env):
    global last_episode, paths, path_idx, last_start, stuck, blocked_nodes
    last_episode = env.episode_account # Initialization of the episode number
    paths.clear()
    path_idx.clear()
    last_start.clear()
    stuck.clear()
    blocked_nodes.clear()
    visited_nodes.clear()
    visited_nodes_twice.clear()

    ## A* computation for every agent
    for agent in range(env.agent_num):
        paths[agent] = a_star(env, env.current_start[agent], env.goal_array[agent],blocked=blocked_nodes)
        path_idx[agent] = 0
        last_start[agent] = env.current_start[agent]
        stuck[agent] = 0

def update_stuck(env):
    """Detect an agent taht is stuck for time > STUCKTHRESHOLD and replan path"""
    for agent in range(env.agent_num):
        if env.current_goal[agent] is not None: ## Agent on node, cant be stuck
            continue
        current = env.current_start[agent]
        if current == env.goal_array[agent]: ## Not stuck, agent arrived.
            stuck[agent] = 0
            last_start[agent] = current
            blocked_nodes.add(current)
            continue

        ## We track the nodes visited by an agent, an consider a deadlock if an agent visit the same node more than twice without reaching its goal.
        if current != last_start[agent]:
            if agent not in visited_nodes:
                visited_nodes[agent] = set()
                visited_nodes[agent].add(current)
            elif current not in visited_nodes[agent]:
                visited_nodes[agent].add(current)
            else: 
                if agent not in visited_nodes_twice:
                    visited_nodes_twice[agent] = set()
                    visited_nodes_twice[agent].add(current)
                elif current not in visited_nodes_twice[agent]:
                    visited_nodes_twice[agent].add(current)
            
        if current == last_start[agent]:
            stuck[agent] += 1 ## Incresing stuck count
        else:
            stuck[agent] = 0
            last_start[agent] = current

        if stuck[agent] > STUCK_THREESHOLD: # Path replan
            stuck[agent] = 0 
            
            others = set(env.current_start[j] for j in range(env.agent_num) if j != agent) | visited_nodes_twice.get(agent, set()) | blocked_nodes # We consider as blocked the nodes currently occuped by other agents, the nodes visited at least twice by the agent and the nodes already reached as goal by other agents

            
            paths[agent] = a_star(env, current, env.goal_array[agent], blocked=others)
            path_idx[agent] = 0






def sync_path(i, env):
    
    current = env.current_start[i]
    path    = paths[i]
    current_place     = path_idx[i]

    if current_place < len(path) and path[current_place] == current:
        return                                     # already in sync
    if current_place + 1 < len(path) and path[current_place + 1] == current:
        path_idx[i] = current_place + 1                    # advanced one step
        return
    # Off path -> replan from current position
    paths[i]    = a_star(env, current, env.goal_array[i],blocked=blocked_nodes)
    path_idx[i] = 0


def find_next_node(agent, env):
    sync_path(agent, env)

    current = env.current_start[agent]
    goal = env.goal_array[agent]

    if current == goal: ## No travel case
        blocked_nodes.add(current)
        return current
    
    path = paths[agent] # We take the full astar path for the agent
    current_place = path_idx[agent]

    if current_place + 1 < len(path): # Agent not finish its way
        return path[current_place + 1] # Return next node in path

    return current # End of path case

# Agent runing toward each other will wait
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
                        actions[i] = start_i
                        do = True
                        break
    return actions

def resolve_conflicts(env, actions):
    """Priority-based conflict resolution.
    Two conflicts are resolved, vertex and swap. 
    Lower-priority agent stays at current node.
    """
    def dist_to_goal(i):
        p   = env.pos
        c, g = env.current_start[i], env.goal_array[i]
        return math.sqrt((p[c][0] - p[g][0]) ** 2 + (p[c][1] - p[g][1]) ** 2)

    order     = sorted(range(env.agent_num), key=dist_to_goal)
    committed = {}   # agent_id -> node they will occupy next step

    # Mid-edge agents are committed to their current_goal (can't stop)
    for i in range(env.agent_num):
        if env.current_goal[i] is not None:
            committed[i] = env.current_goal[i]

    for i in order:
        if env.current_goal[i] is not None:
            continue   # already committed above

        target  = actions[i]
        current = env.current_start[i]

        if target == current:          # agent already chose to wait
            committed[i] = current
            continue

        # Vertex conflict: a higher-priority agent is heading to target
        vertex = any(v == target for k, v in committed.items() if k != i)

        # Swap conflict: a committed agent sits at target and heads to current
        swap = any(
            env.current_start[k] == target and v == current
            for k, v in committed.items()
        )

        if vertex or swap:
            actions[i] = current      # wait
            committed[i] = current
        else:
            committed[i] = target

    return actions

def policy(obs, env):
    global last_episode

    # Reinitialisation at the begening of an episode
    if (env.episode_account != last_episode):
        init(env)
    else:
        update_stuck(env)

    # Giving each agent an action
    actions = []

    for agent in range(env.agent_num):
        if env.current_goal[agent] is not None: #Agent on an edge
            actions.append(env.current_goal[agent])
        else:
            actions.append(find_next_node(agent,env))


    # Safety mechanisms

    # Priority based conflicts resolution mechanism
    actions = resolve_conflicts(env,actions)

    # Colision avoider (Last net)
    actions = apply_safety(env, actions)

    #print(blocked_nodes)


    return actions


    
