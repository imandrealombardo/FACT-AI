import os

datasets = ["compas", "german", "drug"]
attacks =  ["IAF", "RAA", "NRAA", "Koh", "Solans"]
stopping_methods = ["Fairness", "Accuracy"]

# ================= To replicate Figure 2 of the original paper: ================= 

lamb = 1
epsilon = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]

for dataset in datasets:
    for attack in attacks:
        for stopping_method in stopping_methods:
            for eps in epsilon:
                print("\n============================================")
                print("Reproducing the results of Figure 2...")
                print(f"Run parameters: \n - Dataset: {dataset}\n - Attack: {attack}\n - Stopping method: {stopping_method}\n - Epsilon: {eps}, Lambda: {lamb}")
                print("============================================")
                
                os.system(f"python -u run_gradient_em_attack.py --total_grad_iter 10000 --dataset {dataset} --use_slab --epsilon {eps} --lamb {lamb} --method {attack} --stopping_method {stopping_method} --stop_after 2\n")

# ================= To replicate Figure 3 of the original paper: =================

lambdas = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0]
epsilon = [0.1, 0.5, 1.0]

for dataset in datasets:
    for attack in attacks:
        for stopping_method in stopping_methods:
            for eps in epsilon:
                for lamb in lambdas:
                    print("\n============================================")
                    print("Reproducing the results of Figure 3...")
                    print(f"Run parameters: \n - Dataset: {dataset}\n - Attack: {attack}\n - Stopping method: {stopping_method}\n - Epsilon: {eps}, Lambda: {lamb}")
                    print("============================================")
                    
                    os.system(f"python -u run_gradient_em_attack.py --total_grad_iter 10000 --dataset {dataset} --use_slab --epsilon {eps} --lamb {lamb} --method {attack} --stopping_method {stopping_method} --stop_after 2\n")