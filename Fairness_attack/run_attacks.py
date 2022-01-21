import os

# IAF
os.system('coverage run run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.5 --method IAF --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz')
# os.system('coverage html --title=IAF_attack')
# os.system("mv '/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Fairness_attack/htmlcov' '/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Fairness_attack/IAF'")

# RAA
os.system('coverage run run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.5 --method IAF --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz')
# os.system('coverage html --title=RAA_attack')
# os.system("mv '/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Fairness_attack/htmlcov' '/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Fairness_attack/RAA'")

# NRAA
os.system('coverage run run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.5 --method IAF --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz')
# os.system('coverage html --title=NRAA_attack')
# os.system("mv '/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Fairness_attack/htmlcov' '/Users/matteo/Dropbox/Università/Module 3/FACT/Project/FACT-AI/Fairness_attack/NRAA'")
