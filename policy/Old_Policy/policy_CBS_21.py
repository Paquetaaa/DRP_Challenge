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
cbs_success = True  # False if CBS failed and we fell back to A*


WAIT_COST = 1.0  # Coût d'attente pour 1 step (équivalent à la vitesse par défaut)
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

def get_nodes_within_radius(env, center_node, radius):
    res = []
    cx, cy = env.pos[center_node]
    for n in env.G.nodes():
        x, y = env.pos[n]
        if math.hypot(x - cx, y - cy) < radius:
            res.append(n)
    return res

def compute_sic(env, solution):
    sic = 0
    for agent in solution:
        path = solution[agent]
        goal = env.goal_array[agent]
        for t in range(len(path) - 1):
            if path[t] != path[t+1]:
                sic += 1
            else:
                sic += WAIT_COST
            if path[t+1] == goal:
                break # On arrête de compter une fois le but atteint
    return sic

# Utility functions for detect_conflict
def get_node_at(solution, agent_id, t):
    if agent_id not in solution:
        return None
    path = solution[agent_id]
    if t >= len(path):
        return path[-1]
    return path[t]

def reshape_graph_from_G(env, G, pos):
    speed = env.speed
    G_new = nx.DiGraph()
    pos_new = dict(pos)

    def interpolate(p1, p2, alpha):
        return (
            round(p1[0] + alpha * (p2[0] - p1[0]),4),
            round(p1[1] + alpha * (p2[1] - p1[1]),4)
        )

    # Copy of original nodes
    for n in G.nodes():
        G_new.add_node(n, type="original")

    for u, v, data in G.edges(data=True):
        w = data['weight'] # distance between u and v
        k = math.ceil((w / speed) - 1e-9) # Number of intermediate nodes needed to ensure edge traversal takes at least 1 step

        if k == 1: # No intermediate nodes needed, just copy the edge
            G_new.add_edge(u, v, weight=1)
            G_new.add_edge(v, u, weight=1)
            continue

        prev = u
        for i in range(1, k): # Create intermediate nodes, if k = 3 we will create 2 intermediate nodes at 1/3 and 2/3 of the way
            new_node = f"{u}_{v}_{i}"

            G_new.add_node(new_node, type="intermediate")

            alpha = i / k
            pos_new[new_node] = interpolate(pos_new[u],pos_new[v], alpha)
            #print(f"Creating intermediate node {new_node} between {u} and {v} at position {pos_new[new_node]}")

            G_new.add_edge(prev, new_node, weight=1)
            prev = new_node

        G_new.add_edge(prev, v, weight=1)

        prev = v
        for i in range(1, k): # Create intermediate nodes, if k = 3 we will create 2 intermediate nodes at 1/3 and 2/3 of the way
            new_node = f"{v}_{u}_{i}"

            G_new.add_node(new_node, type="intermediate")

            alpha = i / k
            pos_new[new_node] = interpolate(pos_new[v],pos_new[u], alpha)
            #print(f"Creating intermediate node {new_node} between {v} and {u} at position {pos_new[new_node]}")

            G_new.add_edge(prev, new_node, weight=1)
            prev = new_node

        G_new.add_edge(prev, u, weight=1)





    env.pos = pos_new
    return G_new

def get_goal_occupancy_constraints(solution, env, exclude_agent=None):
    constraints = set()

    for agent, path in solution.items():
        if agent == exclude_agent:
            continue

        goal = env.goal_array[agent]

        for t in range(len(path)):
            if path[t] == goal:
                t_goal = t
                break
        else:
            continue

        for future_t in range(t_goal, t_goal + 50):
            constraints.add((agent, goal, future_t))

    return constraints

def a_star_constrained(env, agent, start, goal, constraints):
    """A* pathfinding in env.G using real time (cumulated edge weights) as constraint timestamps."""

    h = h_euclidian(env.pos, start, goal)
    counter = 0
    open_list = [(h, counter, start, 0, [start], 0)]  # (f, counter, node, real_t, path, g)
    visited = set()
    max_edge_w = max((env.G[u][v]['weight'] for u, v in env.G.edges()), default=WAIT_COST)
    max_real_t = int(len(list(env.G.nodes())) * max_edge_w * env.agent_num * 2)

    goal_constraints_times = [c[2] for c in constraints if c[0] == agent and c[1] == goal]
    min_goal_arrival = max(goal_constraints_times) + 1 if goal_constraints_times else 0

    while open_list:
        f, _, node, real_t, path, g = heapq.heappop(open_list)
        if node == goal and real_t >= min_goal_arrival:
            return path
        if (node, real_t) in visited:
            continue
        visited.add((node, real_t))

        for neighbor in env.G.neighbors(node):
            edge_w = round(env.G[node][neighbor]['weight'])
            new_real_t = real_t + edge_w
            if (agent, neighbor, new_real_t) not in constraints and new_real_t <= max_real_t:
                counter += 1
                new_g = g + env.G[node][neighbor]['weight']
                new_f = new_g + h_euclidian(env.pos, neighbor, goal)
                heapq.heappush(open_list, (new_f, counter, neighbor, new_real_t, path + [neighbor], new_g))

        if env.G.nodes[node]['type'] == 'original': # Only allow waiting at original nodes

            wait_real_t = real_t + round(WAIT_COST)
            if (agent, node, wait_real_t) not in constraints and wait_real_t <= max_real_t:
                counter += 1
                new_g = g + WAIT_COST
                new_f = new_g + h_euclidian(env.pos, node, goal)
                heapq.heappush(open_list, (new_f, counter, node, wait_real_t, path + [node], new_g))

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

def detect_conflict(solution, env):
    agents = list(solution.keys())

    for i_idx, i in enumerate(agents):
        path_i = solution[i]
        for j in agents[i_idx+1:]:
            path_j = solution[j]

            max_len = max(len(path_i), len(path_j))

            for t in range(max_len):
                ni = path_i[t] if t < len(path_i) else path_i[-1]
                nj = path_j[t] if t < len(path_j) else path_j[-1]

                # distance réelle
                if h_euclidian(env.pos, ni, nj) < env.speed:
                    return ('proximity', i, j, ni, nj, t)

            # edge swap classique (important)
            for t in range(max_len - 1):
                ni = path_i[t] if t < len(path_i) else path_i[-1]
                nj = path_j[t] if t < len(path_j) else path_j[-1]

                ni_next = path_i[t+1] if t+1 < len(path_i) else path_i[-1]
                nj_next = path_j[t+1] if t+1 < len(path_j) else path_j[-1]

                if ni == nj_next and nj == ni_next:
                    return ('edge', i, j, ni, nj, t)

    return None

def detect_conflict_2(solution,env):
    agents = list(solution.keys())
    for i_idx, i in enumerate(agents):
        path_i = solution[i]
        for j in agents[i_idx+1:]:
            path_j = solution[j]

            max_len = max(len(path_i), len(path_j)) 
            # Only check vertex conflicts at original nodes, edge conflicts are implicitly handled by the fact that intermediate nodes are unique to each edge and can't be shared without a vertex conflict at an original node.
            for ki in range(max_len):
                pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
                pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]

                if (pos_i == pos_j and env.G.nodes[pos_i]['type'] == 'original' and env.G.nodes[pos_j]['type'] == 'original'): # Only consider vertex conflicts at original nodes
                    print(f"Vertex conflict detected between agent {i} and agent {j} at node {pos_i} at time {ki}")
                    return ('vertex', i, j, pos_i, ki, ki)
                if (h_euclidian(env.pos,pos_i,pos_j) < env.speed and (env.G.nodes[pos_i]['type'] == 'original' or env.G.nodes[pos_j]['type'] == 'original')):
                    print(f"Proximity conflict detected between agent {i} and agent {j} at node {pos_i} at time {ki}")
                    return ('vertex', i, j, pos_i, ki, ki)            


            for ki in range(max_len - 1):
                pos_i = path_i[ki] if ki < len(path_i) else path_i[-1]
                pos_j = path_j[ki] if ki < len(path_j) else path_j[-1]
                pos_i_next = path_i[ki+1] if ki+1 < len(path_i) else path_i[-1]
                pos_j_next = path_j[ki+1] if ki+1 < len(path_j) else path_j[-1]

                # Edge conflicts: opposite traversal of the same edge, overlapping times
                if (pos_i == pos_j_next and pos_j == pos_i_next): 
                    return ('edge', i, j, pos_i, pos_j, ki, ki)

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
    root = make_root_node(env)
    open_list = []
    counter = 0

    heapq.heappush(open_list, (root.cost, counter, root))

    while open_list:
        _, _, node = heapq.heappop(open_list)
        solution = node.solution

        # 🔥 AJOUT CRUCIAL
        goal_constraints = get_goal_occupancy_constraints(solution, env)
        node.constraints = node.constraints | goal_constraints

        conflict = detect_conflict(solution, env)
        if conflict is None:
            return solution

        if conflict[0] == 'proximity':
            _, a1, a2, n1, n2, t = conflict

            # 🔥 clé : zone interdite
            forbidden_1 = get_nodes_within_radius(env, n2, env.speed)
            forbidden_2 = get_nodes_within_radius(env, n1, env.speed)

            constraints_1 = node.constraints | {(a1, x, t) for x in forbidden_1}
            constraints_2 = node.constraints | {(a2, x, t) for x in forbidden_2}

        else:  # edge
            _, a1, a2, u, v, t = conflict
            constraints_1 = node.constraints | {(a1, v, t+1)}
            constraints_2 = node.constraints | {(a2, u, t+1)}

        # CHILD 1
        new_sol1 = dict(node.solution)
        constraints_1 |= get_goal_occupancy_constraints(node.solution, env, exclude_agent=a1)
        new_sol1[a1] = a_star_constrained(
            env, a1, env.current_start[a1], env.goal_array[a1], constraints_1
        )
        cost1 = compute_sic(env, new_sol1)
        heapq.heappush(open_list, (cost1, counter := counter+1,
                                  CT_Node(constraints_1, new_sol1, cost1)))

        # CHILD 2
        new_sol2 = dict(node.solution)
        constraints_2 |= get_goal_occupancy_constraints(node.solution, env, exclude_agent=a2)
        new_sol2[a2] = a_star_constrained(
            env, a2, env.current_start[a2], env.goal_array[a2], constraints_2
        )
        cost2 = compute_sic(env, new_sol2)
        heapq.heappush(open_list, (cost2, counter := counter+1,
                                  CT_Node(constraints_2, new_sol2, cost2)))

    return None

def path_translation(env,result):
    """Translate paths from the expanded graph back to the original graph."""
    translated_path = {}
    for agent, path in result.items():
        translated_path[agent] = [node for node in path if not isinstance(node, str)]
    return translated_path


def init(env):
    global last_episode, paths, path_idx, last_node, cbs_success
    last_episode = env.episode_account # Initialization of the episode number
    paths.clear()
    path_idx.clear()
    last_node.clear()

    if not hasattr(env, "G_original"):
        env.G_original = env.G.copy()
        env.pos_original = dict(env.pos)

    env.G = reshape_graph_from_G(env, env.G_original, env.pos_original)

    ## CBS
    result = cbs(env)
    if result is not None:
        print("resultat de CBS : ")
        print(result)
        paths = path_translation(env,result)
        print("resultat du path_translation")
        print(paths)
   
        cbs_success = True
        print("CBS found a solution.")

    else:
        # Fallback: Standard Priority-based A* (more robust than simple A*)
        cbs_success = False
        print("CBS failed to find a solution, falling back to priority-based A*.")
        blocked = set()
        for agent in range(env.agent_num):
            p = a_star_constrained(env, agent, env.current_start[agent], env.goal_array[agent],
                                   {(agent, b, t) for b in blocked for t in range(100)})
            
            paths[agent] = p
            blocked.add(env.goal_array[agent])

        paths = path_translation(env,paths)
        
    for agent in range(env.agent_num):
        path_idx[agent] = 0
        last_node[agent] = env.current_start[agent]

def policy(obs, env):
    global last_episode

    if env.episode_account != last_episode or len(paths) != env.agent_num:
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

    return actions
