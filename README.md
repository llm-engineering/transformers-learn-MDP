# Transformers-Learn-MDP-Transitions
This is the codebase for the paper *Transformers Learn Transition Dynamics when Trained to Predict Markov Decision Processes*.

Through the code above, we achieve the process of training and testing the probes used in the experiment. The exact process is outlined as follows:

## Experiment Design

The steps of the experiment are:

#### Gridworld

1. Generate training data for the transformer by playing through Gridworld (in *RL_Training_Gridworld*)
2. Train transformers on generated training data (in *GPT/GridWorld*)
3. Generate embeddings using transformers (in *Probe*)
4. Train probes on embeddings and collect data (in *Probe/probe.py*)

#### ConnectFour

1. Generate training data for the transformer by playing through Gridworld (in *RL_Training_ConnectFour*)
2. Train transformers on generated training data (in *transformer_training* and *transformer_training_mcts*)
3. Generate embeddings using transformers (in *transformers_trained* and *transformers_trained_mcts*)
4. Train probes on embeddings and collect data (in *Probe_training*)
5. Parse data (in *parse_probe_data.py*)


