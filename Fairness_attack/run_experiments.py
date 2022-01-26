import os

epsilons = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
lambdas =  [0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
datasets = ["compas"]                # ["german", "compas", "drug"]
attacks =  ["IAF"]                   # ["IAF", "RAA", "RNAA", "Koh", "Solans"]
stopping_methods = ["Parity"]        # ["Parity", "Accuracy"]


for eps in epsilons: 
    for lamb in lambdas: 
        for dataset in datasets:
            for attack in attacks:
                for stopping_method in stopping_methods:

                    # Print run information in the terminal
                    print("\n============================================")
                    print(f"Run parameters: \n - Dataset: {dataset}\n - Attack: {attack}\n - Stopping method: {stopping_method}\n - Epsilon: {eps}, Lambda: {lamb}")
                    print("============================================")
                    
                    os.system(f"python -u run_gradient_em_attack.py --total_grad_iter 10000 --dataset {dataset} --use_slab --epsilon {eps} --lamb {lamb} --method {attack} --sensitive_attr_filename {dataset}_group_label.npz --stopping_method {stopping_methods} --stop_after 5\n")

