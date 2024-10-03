import re
import numpy as np
from collections import defaultdict

from collections import defaultdict

def parse_winrate_file(file_path, output_path):
    data = defaultdict(list)

    pattern = re.compile(r"Winrate for probe mode (\d+) seed \d+ layer (\d+) linear (\w+) mcts (\w+): ([\d\.]+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                mode = int(match.group(1))
                layer = int(match.group(2))
                linear = match.group(3) == 'True'
                mcts = match.group(4) == 'True'
                winrate = float(match.group(5))

                # Group winrates by (mode, layer, linear, mcts)
                key = (mode, layer, linear, mcts)
                data[key].append(winrate)

    with open(output_path, 'w') as file:

        for key, winrates in data.items():
            mode, layer, linear, mcts = key
            mean_winrate = np.mean(winrates)
            std_winrate = np.std(winrates, ddof=1)  
            file.write(f"Mode {mode} Layer {layer} Linear {linear} MCTS {mcts} - Mean: {mean_winrate:.4f}, Std: {std_winrate:.4f}\n")
    
    data = defaultdict(list)

    pattern = re.compile(r"Winrate for probe mode \d+ seed \d+ layer \d+ linear (\w+) mcts (\w+): ([\d\.]+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                linear = match.group(1) == 'True'
                mcts = match.group(2) == 'True'
                winrate = float(match.group(3))
                key = (linear, mcts)
                data[key].append(winrate)

    with open(output_path, 'a') as file:
        for key, winrates in data.items():
            linear, mcts = key
            mean_winrate = np.mean(winrates)
            std_winrate = np.std(winrates, ddof=1)  
            file.write(f"Linear {linear} MCTS {mcts} - Mean: {mean_winrate:.4f}, Std: {std_winrate:.4f}\n")

def parse_winrate_file_random(file_path, output_path):
    data = defaultdict(list)

    pattern = re.compile(r"Winrate for random probe seed \d+ linear (\w+) mcts (\w+): ([\d\.]+)")

    with open(file_path, 'r') as file:
        for line in file:
            match = pattern.match(line.strip())
            if match:
                linear = match.group(1) == 'True'
                mcts = match.group(2) == 'True'
                winrate = float(match.group(3))
                key = (linear, mcts)
                data[key].append(winrate)

    with open(output_path, 'w') as file:

        for key, winrates in data.items():
            linear, mcts = key
            mean_winrate = np.mean(winrates)
            std_winrate = np.std(winrates, ddof=1)  
            file.write(f"Linear {linear} MCTS {mcts} - Mean: {mean_winrate:.4f}, Std: {std_winrate:.4f}\n")

def main():
    parse_winrate_file('decision_outputs.txt', 'decision_outputs_summarized.txt')
    parse_winrate_file_random('decision_outputs_random.txt', 'decision_outputs_summarized_random.txt')
    parse_winrate_file('decision_outputs_vs_mcts.txt', 'decision_outputs_summarized_vs_mcts.txt')
    parse_winrate_file_random('decision_outputs_random_vs_mcts.txt', 'decision_outputs_summarized_random_vs_mcts.txt')

if __name__ == "__main__":
    main()