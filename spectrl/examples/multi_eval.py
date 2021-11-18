from spectrl.util.io import parse_command_line_options, read_multi_stats

import numpy as np

if __name__ == '__main__':
    flags = parse_command_line_options()
    itno = flags['itno']
    folder = flags['folder']

    avg_probs, min_eps, samples = read_multi_stats(itno, folder)
    avg_welfare = np.mean(avg_probs)
    std_welfare = np.std(avg_probs)
    avg_eps = np.mean(min_eps)
    std_eps = np.std(min_eps)
    num_runs = len(avg_probs)
    avg_samples = np.mean(samples)
    print('Average social welfare: {} (std = {})'.format(avg_welfare, std_welfare))
    print('Average epsilon: {} (std = {})'.format(avg_eps, std_eps))
    print('Number of successful runs: {}'.format(num_runs))
    print('Average samples: {}'.format(avg_samples))

    print('\nLatex: {:.2f} \\textpm ~ {:.2f} & {:.2f} \\textpm ~ {:.2f} & {} & {:.2f}'.format(
        avg_welfare, std_welfare, avg_eps, std_eps, num_runs, avg_samples/1000000
    ))
