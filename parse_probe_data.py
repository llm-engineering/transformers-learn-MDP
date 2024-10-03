import argparse
import re
import numpy as np
from collections import defaultdict

def parse_probe_data(input_file_path, output_file_path, t):
    substrings = []
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = re.split(r'(Layer)', line.strip())
            for i in range(1, len(parts), 2):
                substring = parts[i] + parts[i + 1]
                substrings.append(substring)

    last_occurrence = {}

    prefix_pattern = re.compile(r"^(Layer \d+ Seed \d+ Mode \d+ losses:)")

    #print(substrings)
    for line in substrings:
        match = prefix_pattern.match(line)
        if match:
            prefix = match.group(1)
            last_occurrence[prefix] = line.strip()

    data = defaultdict(lambda: {'a': [], 'b': []})

    pattern = re.compile(r"Layer (\d+) Seed \d+ Mode (\d+) losses: ([\d\.]+),\s*([\d\.]+)")

    for line in last_occurrence.values():
        match = pattern.match(line.strip())
        if match:
            layer = int(match.group(1))
            mode = int(match.group(2))
            a = float(match.group(3))
            b = float(match.group(4))

            data[(layer, mode)]['a'].append(a)
            data[(layer, mode)]['b'].append(b)
        else:
            print(line)
    #print(data.items())
    
    with open(output_file_path, 'w') as output_file:
        for (layer, mode), values in data.items():
            mean_a = np.mean(values['a'])
            std_a = np.std(values['a'], ddof=1) 
            mean_b = np.mean(values['b'])
            std_b = np.std(values['b'], ddof=1)

            output_file.write(f"Layer {layer} Mode {mode} MCTS {t} Linear: Mean={mean_a:.6f}, Std={std_a:.6f}, "
                f"Nonlinear: Mean={mean_b:.6f}, Std={std_b:.6f}\n")
            
def main():
    for t in [True, False]:
        for m in range(2):
            if t:
                input_file_path = rf'transformers_trained_mcts/mcts_mode{m}/probe_data/test_losses.txt'
                output_file_path = rf'transformers_trained_mcts/mcts_mode{m}/probe_data/test_losses_parsed_mode_{m}_mcts_{t}.txt'
            else:
                input_file_path = rf'transformers_trained/RL_mode{m}/probe_data/test_losses.txt'
                output_file_path = rf'transformers_trained/RL_mode{m}/probe_data/test_losses_parsed_mode_{m}_mcts_{t}.txt'
            parse_probe_data(input_file_path, output_file_path, t)
            if t:
                input_file_path = rf'transformers_trained_mcts/mcts_mode{m}/probe_data/test_losses_mse.txt'
                output_file_path = rf'transformers_trained_mcts/mcts_mode{m}/probe_data/test_losses_parsed_mode_{m}_mcts_{t}_mse.txt'
            else:
                input_file_path = rf'transformers_trained/RL_mode{m}/probe_data/test_losses_mse.txt'
                output_file_path = rf'transformers_trained/RL_mode{m}/probe_data/test_losses_parsed_mode_{m}_mcts_{t}_mse.txt'
            parse_probe_data(input_file_path, output_file_path, t)

if __name__ == "__main__":
    main()