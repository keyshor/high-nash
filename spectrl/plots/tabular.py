from spectrl.util.io import parse_command_line_options, read_multi_stats

import numpy as np

precision = 2

if __name__ == '__main__':
    flags = parse_command_line_options()
    itno = flags['itno']
    folder = flags['folder']
    spec = flags['spec_num']
    algo_list = ['search/', 'nvi/', 'multi_qrm/']
    # algo_list = ['search/', 'multi_qrm']

    for spec_num in range(spec+1):
        trow = ['spec'+str(spec_num)]

        for algo in algo_list:

            folder_name = folder+'/spec' + str(spec_num) + '/' + algo
            avg_probs, min_eps = read_multi_stats(itno, folder_name)

            sw = [np.round(np.mean(avg_probs), precision), np.round(np.std(avg_probs), precision)]
            sw_str = '{} \\textpm ~ {}'.format(sw[0], sw[1])
            trow.append(sw_str)

            ep = [np.round(np.mean(min_eps), precision), np.round(np.std(min_eps), precision)]
            ep_str = '{} \\textpm ~ {}'.format(ep[0], ep[1])
            trow.append(ep_str)

            runs = len(avg_probs)
            if runs == 'nan':
                trow.append('0')
            else:
                trow.append(str(runs))

        print((' & '.join(trow)).replace('nan', '--') + '  \\\\ \\hline ')
