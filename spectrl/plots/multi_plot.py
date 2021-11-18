import os
from matplotlib import pyplot as plt
from spectrl.util.io import plot_error_bar, extract_plot_data, save_plot, parse_command_line_options

'''
0: Samples
1: Time
-1: Mean Probability
'''

metric = -1

x_index = 0
start_itno = 0
x_max = 1e4

flags = parse_command_line_options()
folder = flags['folder']
itno = flags['itno']
spec = flags['spec_num']
folder = os.path.join(folder, 'spec{}'.format(spec))

search_folder = os.path.join(folder, "search")
qrm_folder = os.path.join(folder, "multi_qrm")
nvi_folder = os.path.join(folder, "nvi")

xs, _, _, _ = extract_plot_data(search_folder, x_index, start_itno, itno)
ys = extract_plot_data(search_folder, metric, start_itno, itno)
plot_error_bar(xs, ys, 'blue', 'Search (Ours)', points=True)
x_max = max(x_max, xs[-1]+3000)


xq, _, _, _ = extract_plot_data(qrm_folder, x_index, start_itno, itno)
yq = extract_plot_data(qrm_folder, metric, start_itno, itno)
plot_error_bar(xq, yq, 'seagreen', 'Multi-QRM')
x_max = max(x_max, xq[-1])

'''
xt, _, _, _ = extract_plot_data(nvi_folder, x_index, start_itno, itno)
yt = extract_plot_data(nvi_folder, metric, start_itno, itno)
plot_error_bar(xt, yt, 'tomato', 'NVI', points=True)
x_max = min(x_max, xt[-1])
'''

plt.xlim(right=x_max, left=-0.01)

save_folder = 'spectrl/plots'
plot_name = folder.replace("/","_")  
save_plot(save_folder, plot_name, False)
