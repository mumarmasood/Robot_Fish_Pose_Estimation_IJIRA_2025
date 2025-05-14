import csv
import os
import matplotlib.pyplot as plt
import numpy as np

# Set font and rendering options
plt.rcParams.update({
    'font.family': 'serif',
    'mathtext.fontset': 'cm',   # Computer Modern
    # 'font.serif': ['Times New Roman'],
    'font.size': 16,
    'axes.labelsize': 16,
    'axes.titlesize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    # 'figure.facecolor': 'black'
})

# Plot both signals with labeled PID info
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

path = 'Sim_Data/v3/'
files_list = [
    'Sim_False_kp2_ki0.1_kd1_rc1_200filt.csv',
    'Sim_False_kp3_ki0.1_kd1_rc1_200filt.csv',
    'Sim_False_kp3_ki0.5_kd1_rc1_200filt.csv',
    'Sim_False_kp3_ki0.5_kd5_rc1_200filt.csv'
]


for file in files_list:
    # Extract PID values from filename
    parts = file.replace('.csv', '').split('_')
    kp = parts[2][2:]
    ki = parts[3][2:]
    kd = parts[4][2:]

    label_base = f'$k_p={kp}, k_i={ki}, k_d={kd}$'

    time = []
    delta = []
    psi = []

    
    with open(os.path.join(path, file), 'r') as csvfile:
        reader = csv.reader(csvfile)
        header = next(reader)  # Skip header
        for row in reader:
            if not row or len(row) < 5:  # Skip empty or short rows
                continue
            try:
                time.append(float(row[0]))
                psi.append(float(row[3]))
                delta.append(float(row[4]))
            except ValueError:
                continue  # Skip rows that can't be parsed as floats

        ax1.plot(time, delta, label={label_base})
        ax2.plot(time, psi, label={label_base})
    # Apply avg filter to delta_hist
    # filter_size = 15
    # time = time[:-filter_size+1]
    # delta = np.convolve(delta, np.ones(filter_size)/filter_size, mode='valid')
    # psi = np.convolve(psi, np.ones(filter_size)/filter_size, mode='valid')


    


# Final formatting




plt.xlabel(r'Time (s)')
plt.suptitle('IBVS Tracking Performance Across PID Gains')
ax1.set_ylabel(r'Control Signal $\delta$ (degrees)')
ax2.set_ylabel(r'Tracking Error $\psi_e$ (degrees)')
ax1.grid(True)
ax2.grid(True)
# ax1.legend(loc='upper right')
ax2.legend(loc='lower right')
plt.tight_layout()

plt.xlim(0, 35)
plt.show()



# # another figure
# fig1, ax3 = plt.subplots(figsize=(10, 4))

# ax3.plot(time, delta, label='Control Signal $(\delta)$')

# plt.show()