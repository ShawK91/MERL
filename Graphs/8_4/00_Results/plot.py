import numpy as np, os, argparse
from matplotlib import pyplot as plt
import seaborn as sns; sns.set(color_codes=True)


TOTAL_RUNS = 1000000
sampling = 1
bucket = 4000

class Frame:
	def __init__(self, label, x, y, color, linestyle):
		self.y = y.transpose()
		self.x = x.transpose()
		self.label = label
		self.color = color
		self.linestyle = linestyle

def get_data():

	all_folders = [name for name in os.listdir(".") if os.path.isdir(name)]

	data_dict = {}

	for folder in all_folders:

		all_files = [folder + '/'+name for name in os.listdir(folder) if os.path.isfile(folder + '/'+name)]
		#data = [[] for i in range(int(TOTAL_RUNS))]

		# Get all data
		all_data = []
		for filename in all_files:  # Each statistical run
			try:
				data = np.loadtxt(filename, delimiter=',')
			except:
				continue

			# Parse to keep under total_runs
			mask = data[:, 0] < TOTAL_RUNS
			mask = np.reshape(mask, (len(mask), 1))
			data = np.multiply(data, mask) + np.multiply(np.zeros((len(mask), 1)) - 1, (1 - mask))

			# Trim array
			trim_point = -1
			for i, loc in enumerate(data):
				if loc[0] == -1 and loc[1] == -1:
					trim_point = i
					break
			print(data.shape, trim_point)
			data = data[0:trim_point, :]
			print(data.shape)
			all_data.append(data)

		# Bucket data [correct for bias in x]
		all_y = []
		all_x = []
		for i in range(0, TOTAL_RUNS, bucket):
			temp_y = []
			for data in all_data:
				is_relevant = np.logical_and(data[:, 0] > i, data[:, 0] < i + bucket)
				if np.sum(is_relevant) == 0: continue
				y_contrib = np.sum(is_relevant * data[:, 1]) / np.sum(is_relevant)
				temp_y.append(y_contrib)

			if len(temp_y) == 0:
				continue
			else:
				all_x.append(float(i / 1000000.0))
				avg_booster = sum(temp_y) / len(temp_y)
				while len(temp_y) < len(all_data):
					temp_y.append(avg_booster)
					print('BOOST')
				all_y.append(temp_y)

		decorator = np.reshape(np.array(all_x), (len(all_x), 1))
		# Also save raw data
		raw_data = decorator
		raw_data = np.concatenate((raw_data, np.array(all_y)/2.0), axis=1)

		data_dict[folder] = raw_data


	return data_dict


#Get all the raw data out with key as the tag (foldername)
data_dict = get_data()

all_data = []
for tag, data in data_dict.items():
	if tag == 'MERL':
		label = 'MERL'; color = 'r'; linestyle='solid'

	elif tag == 'TD3':
		label = 'TD3'
		color = 'g'; linestyle='dashed'

	elif tag == 'EA':
		label = 'NE'
		color = 'b'; linestyle='dashed'

	# elif tag == 'p13':
	#     label = 'TD3_gamma=0.997'; color = 'y'; linestyle='dashed'
	#
	# elif tag == 'p13':
	#     label = 'TD3_gamma=0.9995'; color = 'c'; linestyle = 'dashed'
	#
	# elif tag == 'ne':
	#     label = 'Neuroevolution'; color = 'k'; linestyle = 'dashdot'




	all_data.append(Frame(label, data[:,0:1], data[:,1::], color, linestyle))



reorder = [1,2,0]
all_data = [all_data[i] for i in reorder] #SImple hack to reorder labels in the order desired

for data in all_data:

	ax = sns.tsplot(data.y, time = data.x, ci=95, color = data.color, linestyle=data.linestyle, condition=data.label)
	#ax = sns.tsplot(data.y, time = data.x, color = data.color, ci = 95, n_boot=10000, err_style='ci_band')
	#ax = sns.tsplot(data.y, time = data.x, err_style="unit_traces", color = data.color)

plt.xlabel('Million Steps', fontsize=16)
plt.ylabel('Performance', fontsize=16)
plt.yticks(fontsize = 15)
plt.xticks(fontsize = 15)
plt.legend(fontsize = 16, loc = 2)
plt.tight_layout()

#plt.show()
ax.get_figure().savefig('loose_3_1.png')
#ax.get_figure().savefig(SAVETAG+".eps")
