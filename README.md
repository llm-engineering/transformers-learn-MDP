# :construction: WIP: Cleanup in progress

Credits to the original authors, this is just a cleanup to make the original code sane to go through and run

# Transformers Learn MDP Transitions
This is the codebase for the paper *Transformers Learn Transition Dynamics when Trained to Predict Markov Decision Processes*.

Through the code above, we achieve the process of training and testing the probes used in the experiment. The exact process is outlined as follows:

## Experiment Design

The steps of the experiment are:

#### ConnectFour

1. Generate training data for the transformer by playing through Gridworld (in *RL_Training_ConnectFour*)
2. Train transformers on generated training data (in *transformer_training* and *transformer_training_mcts*)
3. Generate embeddings using transformers (in *transformers_trained* and *transformers_trained_mcts*)
4. Train probes on embeddings and collect data (in *Probe_training*)
5. Parse data (in *parse_probe_data.py*)


# Citations

```
@inproceedings{chen-etal-2024-transformers,
    title = "Transformers Learn Transition Dynamics when Trained to Predict {M}arkov Decision Processes",
    author = "Chen, Yuxi  and
      Ma, Suwei  and
      Dear, Tony  and
      Chen, Xu",
    editor = "Belinkov, Yonatan  and
      Kim, Najoung  and
      Jumelet, Jaap  and
      Mohebbi, Hosein  and
      Mueller, Aaron  and
      Chen, Hanjie",
    booktitle = "Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.blackboxnlp-1.13/",
    doi = "10.18653/v1/2024.blackboxnlp-1.13",
    pages = "207--216",
    abstract = "Language models have displayed a wide array of capabilities, but the reason for their performance remains a topic of heated debate and investigation. Do these models simply recite the observed training data, or are they able to abstract away surface statistics and learn the underlying processes from which the data was generated? To investigate this question, we explore the capabilities of a GPT model in the context of Markov Decision Processes (MDPs), where the underlying transition dynamics and policies are not directly observed. The model is trained to predict the next state or action without any initial knowledge of the MDPs or the players' policies. Despite this, we present evidence that the model develops emergent representations of the underlying parameters governing the MDPs."
}
```


