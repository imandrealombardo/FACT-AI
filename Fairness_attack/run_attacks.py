import os

os.system('coverage erase')
# IAF
os.system('coverage run -a run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.5 --method IAF --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz')
# RAA
os.system('coverage run -a run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.5 --method IAF --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz')
# NRAA
os.system('coverage run -a run_gradient_em_attack.py --em_iter 0 --total_grad_iter 10000 --dataset german --use_slab --epsilon 0.5 --method IAF --sensitive_feature_idx 36 --sensitive_attr_filename german_group_label.npz')

os.system('coverage html --title=Total_coverage')
