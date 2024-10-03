import argparse
import re
import numpy as np
from collections import defaultdict

def main():
    input_file_path = rf'random_probes/rl/test_losses.txt'
    
    pattern = re.compile(r"Random Seed \d+ losses: ([\d\.]+),\s*([\d\.]+)")
    data = defaultdict(list)
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line)
            match = pattern.match(line.strip())
            if match:
                a = float(match.group(1))
                b = float(match.group(2))

                data['a'].append(a)
                data['b'].append(b)
    

    output_file_path = rf'random_test_loss.txt'
    
    with open(output_file_path, 'w') as output_file:
        mean_a = np.mean(data['a'])
        std_a = np.std(data['a'], ddof=1) 
        mean_b = np.mean(data['b'])
        std_b = np.std(data['b'], ddof=1)

        output_file.write(f"Random Test Loss Linear: Mean={mean_a:.6f}, Std={std_a:.6f}, "
            f"Nonlinear: Mean={mean_b:.6f}, Std={std_b:.6f}\n")
    
    input_file_path = rf'random_probes/mcts/test_losses.txt'
    
    pattern = re.compile(r"Random Seed \d+ losses: ([\d\.]+),\s*([\d\.]+)")
    data = defaultdict(list)
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line)
            match = pattern.match(line.strip())
            if match:
                a = float(match.group(1))
                b = float(match.group(2))

                data['a'].append(a)
                data['b'].append(b)
    

    output_file_path = rf'random_test_loss_mcts.txt'
    
    with open(output_file_path, 'w') as output_file:
        mean_a = np.mean(data['a'])
        std_a = np.std(data['a'], ddof=1) 
        mean_b = np.mean(data['b'])
        std_b = np.std(data['b'], ddof=1)

        output_file.write(f"Random Test Loss Linear: Mean={mean_a:.6f}, Std={std_a:.6f}, "
            f"Nonlinear: Mean={mean_b:.6f}, Std={std_b:.6f}\n")
        
    input_file_path = rf'random_probes/rl/test_losses_mse.txt'
    
    pattern = re.compile(r"Random Seed \d+ losses: ([\d\.]+),\s*([\d\.]+)")
    data = defaultdict(list)
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line)
            match = pattern.match(line.strip())
            if match:
                a = float(match.group(1))
                b = float(match.group(2))

                data['a'].append(a)
                data['b'].append(b)
    

    output_file_path = rf'random_test_loss_mse.txt'
    
    with open(output_file_path, 'w') as output_file:
        mean_a = np.mean(data['a'])
        std_a = np.std(data['a'], ddof=1) 
        mean_b = np.mean(data['b'])
        std_b = np.std(data['b'], ddof=1)

        output_file.write(f"Random Test Loss Linear: Mean={mean_a:.6f}, Std={std_a:.6f}, "
            f"Nonlinear: Mean={mean_b:.6f}, Std={std_b:.6f}\n")
    
    input_file_path = rf'random_probes/mcts/test_losses_mse.txt'
    
    pattern = re.compile(r"Random Seed \d+ losses: ([\d\.]+),\s*([\d\.]+)")
    data = defaultdict(list)
    with open(input_file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            print(line)
            match = pattern.match(line.strip())
            if match:
                a = float(match.group(1))
                b = float(match.group(2))

                data['a'].append(a)
                data['b'].append(b)
            if not match:
                print(line)
                print("what")
    

    output_file_path = rf'random_test_loss_mcts_mse.txt'
    
    with open(output_file_path, 'w') as output_file:
        mean_a = np.mean(data['a'])
        std_a = np.std(data['a'], ddof=1) 
        mean_b = np.mean(data['b'])
        std_b = np.std(data['b'], ddof=1)

        output_file.write(f"Random Test Loss Linear: Mean={mean_a:.6f}, Std={std_a:.6f}, "
            f"Nonlinear: Mean={mean_b:.6f}, Std={std_b:.6f}\n")


if __name__ == "__main__":
    main()