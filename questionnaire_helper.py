import numpy as np
from record_demo_data import argparser

def main():
    args = argparser()  # --sub_id
    experiment_data_dir = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/robot_factor_exp/'
    factor_orders = np.genfromtxt(experiment_data_dir + 'factor_orders_list.csv') # 2d np array in the form of (total_subject_num, total_factor_num)
    factor_order = factor_orders[args.sub_id - 1].astype(int)

    print("[Subject {} Factor Order]: {}".format(args.sub_id, factor_order + 1))


if __name__ == '__main__':
    main()