import gym
import math
import networkx as nx
import heapq

### submission information ####
TEAM_NAME = "GRENECHE Lucas"
##############################

##############################
# Global Variables 
#CBS & A*
paths = {} # Dict, key = Agent_id, value = A* path
last_episode = -1
path_idx = {} # Dict, key = Agent_id, value = current node index in paths
last_node = {} # To detect when an agent has reached its next waypoint
## A*
stuck = {} # Dict, key = Agent_id, value = stepspend stuck
blocked_nodes = set() # Set of nodes reached by agents as their goal
STUCK_THREESHOLD = 1
WAIT_COST = 5  # ex: 10
last_start={}

mode = None
pb_count = 1


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

## A* for CBS
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

## ASTAR    
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
		if current == last_start[agent]:
			stuck[agent] += 1 ## Incresing stuck count
		else:
			stuck[agent] = 0
			last_start[agent] = current

		if stuck[agent] > STUCK_THREESHOLD: # Path replan
			stuck[agent] = 0 
			others = [env.current_start[j] for j in range(env.agent_num) if j != agent] 
			others += blocked_nodes
			
			## others = [env.current_start[j] for j in range(env.agent_num) if j != agent and blocked_nodes] Score increase by 100 points
			paths[agent] = a_star(env, current, env.goal_array[agent], blocked=others)
			path_idx[agent] = 0


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
	solution = {}
	# calculation of a_star path for every agent
	for agent in range(env.agent_num):
		current = env.current_start[agent]
		end = env.goal_array[agent]

		solution[agent] = a_star_constrained(env,agent,current,end,set())
	sic = compute_sic(env,solution)

	return CT_Node(set(),solution,sic)
	

def cbs(env):
	root = make_root_node(env)

	open_list = []
	heapq.heappush(open_list, (root.cost, 0, root))

	counter = 1
	max_iter = 5000   
	iter_count = 0

	while open_list and iter_count < max_iter:
		iter_count += 1

		# Early stopping intelligent
		# if iter_count > 500 and len(open_list) > 500:
		# 	break

		_, _, node = heapq.heappop(open_list)
		solution = node.solution

		conflict = detect_conflict(solution)

		if conflict is None:
			return solution

		conflict_type, a1, a2, node_conflict, t = conflict

		# Construire contraintes
		if conflict_type == 'vertex':
			c1 = (a1, node_conflict, t)
			c2 = (a2, node_conflict, t)
		else:
			node_i = node_conflict
			node_j = get_node_at(solution, a2, t)
			c1 = (a1, node_j, t+1)
			c2 = (a2, node_i, t+1)

		low = max(a1, a2)

		constraint = c1 if low == a1 else c2
		new_constraints = node.constraints | {constraint}

		new_path = a_star_constrained(
			env,
			low,
			env.current_start[low],
			env.goal_array[low],
			new_constraints
		)

		new_solution = dict(solution)
		new_solution[low] = new_path

		new_cost = compute_sic(env, new_solution)

		heapq.heappush(open_list, (new_cost, counter,
			CT_Node(new_constraints, new_solution, new_cost)))
		counter += 1

	return None

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

def choose_mode(env):
	n_nodes = len(env.G.nodes)
	n_agents = env.agent_num
	if n_nodes <= 24 and n_agents <= 8:
		return "CBS"
	else:
		return "ASTAR"



def init_CBS(env):
	global last_episode, paths, path_idx, last_node
	last_episode = env.episode_account # Initialization of the episode number
	paths.clear()
	path_idx.clear()
	last_node.clear()
	blocked_nodes.clear()

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

def init_astar(env):
	global last_episode, paths, path_idx, last_start, stuck, blocked_nodes
	last_episode = env.episode_account # Initialization of the episode number
	paths.clear()
	path_idx.clear()
	last_start.clear()
	stuck.clear()
	blocked_nodes.clear()

	## A* computation for every agent
	for agent in range(env.agent_num):
		paths[agent] = a_star(env, env.current_start[agent], env.goal_array[agent],blocked=blocked_nodes)
		path_idx[agent] = 0
		last_start[agent] = env.current_start[agent]
		stuck[agent] = 0



def policy(obs, env):
	global last_episode, mode

	mode = choose_mode(env)

	if mode == 'ASTAR':
		if (env.episode_account != last_episode):
			init_astar(env)
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


	else:

		if (env.episode_account != last_episode):
			init_CBS(env)

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