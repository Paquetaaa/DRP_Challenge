import os
import matplotlib.pyplot as plt
import json

path1 = "/Users/lucas/Desktop/DRP_challenge/main/GRENECHE Lucas_CBS.json"
path2 = "/Users/lucas/Desktop/DRP_challenge/main/GRENECHE Lucas_A*.json"
path3 = "/Users/lucas/Desktop/DRP_challenge/main/GRENECHE Lucas_A*_CBS_mix.json"

def load_data(path, label_prefix=""):
    with open(path, "r") as f:
        data = json.load(f)
        scores = data["Score"]
        instance_ids = [score["instance_id"] for score in scores]
        subtotal_costs = [score["subtotal_cost"] for score in scores]
        runtimes = [score["runtime"] for score in scores]
        distances = [score["distance"] for score in scores]
        # You can also load other metrics if needed, e.g., runtimes, distances, etc.
    return instance_ids, subtotal_costs, runtimes, distances, label_prefix

def plot_comparison(paths_data):
    
    plt.figure(figsize=(12, 8))
    plt.xlabel('Instance ID')
    plt.ylabel('Distance ')
    plt.title('Distance Comparison by Instance')

    for instance_ids, subtotal_costs, runtimes, distances, label_prefix in paths_data:
        #plt.plot(instance_ids, subtotal_costs, label=f'{label_prefix} Subtotal Cost')
        #plt.plot(instance_ids, runtimes, label=f'{label_prefix} Runtime')
        plt.plot(instance_ids, distances, label=f'{label_prefix} Distance')

    
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    # Load data for CBS
    cbs_instance_ids, cbs_subtotal_costs, cbs_runtimes, cbs_distances, cbs_label = load_data(path1, "CBS")
    
    # Load data for A*
    astar_instance_ids, astar_subtotal_costs, astar_runtimes, astar_distances, astar_label = load_data(path2, "A*")

    # Load data for Mix
    mix_instance_ids, mix_subtotal_costs, mix_runtimes, mix_distances, mix_label = load_data(path3, "Mix")  
    
    # Plot both on the same graph
    plot_comparison([(cbs_instance_ids, cbs_subtotal_costs, cbs_runtimes, cbs_distances, cbs_label), 
                     (astar_instance_ids, astar_subtotal_costs, astar_runtimes, astar_distances, astar_label),
                     (mix_instance_ids, mix_subtotal_costs, mix_runtimes, mix_distances, mix_label)])