import numpy as np
import random


def save_orders_list(saving_dir, orders_list):
    orders_list = np.array(orders_list)

    try:
        with open(saving_dir, 'ab') as f_handle:
            np.savetxt(f_handle, orders_list, fmt='%s')
    except FileNotFoundError:
        with open(saving_dir, 'wb') as f_handle:
            np.savetxt(f_handle, orders_list, fmt='%s')


def main():
    """ Generate orders for Human-feature experiment and Robot-factor experiment """
    sub_total_num = 20
    factor_total_num = 6
    modes_total_num = 2
    factor_orders_list = [] # 2d list, in the form of (total_subject_num, total_factor_num),  for example: [[0,1,2,3,4], [1,0,2,3,4], ...]
    mode_orders_list = [] # 2d list, in the form of (total_subject_num, total_factor_num * total_mode_num), for example: [[0,1, 1,0, 0,1], [1,0, 1,0, 0,1], ..., ]
    dir_factor_orders = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/robot_factor_exp/' + 'factor_orders_list.csv'
    dir_mode_orders = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/robot_factor_exp/' + 'mode_orders_list.csv'

    for sub_id in range(sub_total_num):
        factor_order = random.sample(list(range(factor_total_num)), factor_total_num)
        factor_orders_list.append(factor_order)

        current_mode_order = []
        for factor_id in range(factor_total_num):
            mode_order = random.sample(list(range(modes_total_num)), modes_total_num)
            for order in mode_order:
                current_mode_order.append(order)
        mode_orders_list.append(current_mode_order)

    save_orders_list(dir_factor_orders, factor_orders_list)
    save_orders_list(dir_mode_orders, mode_orders_list)

    print(factor_orders_list)
    print(mode_orders_list)

    """ Generate orders for evaluation experiment """
    sub_total_num = 20
    baseline_total_num = 3
    baseline_orders_list = []
    dir_baseline_orders = '/Users/ullrich/ullrich_ws/Projects/Greeting_Learning/experiment_data/evaluation_exp/' + 'baseline_orders_list.csv'

    for sub_id in range(sub_total_num):
        baseline_order = random.sample(list(range(baseline_total_num)), baseline_total_num)
        baseline_orders_list.append(baseline_order)

    save_orders_list(dir_baseline_orders, baseline_orders_list)
    print(baseline_orders_list)


if __name__ == '__main__':
    main()