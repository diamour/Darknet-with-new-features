import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter  
filename = ["sigResult_se12layer.txt", "weights_fc1_se12layer.txt", "weights_fc2_se12layer.txt"]
filename_sig = ["sigResult_se1024layer.txt", "sigResult_se2048layer.txt", "sigResult_se4096layer.txt"]
filename_weights = ["weights_se15layer.txt", "weights_se13layer.txt", "weights_se21layer.txt"]
filename_mask = ["maskResult_4layer.txt", "maskResult_2layer.txt"]
filename_biases = ["biaes_se13layer.txt", "biaes_se15layer.txt", "biaes_se21layer.txt"]

filenum = len(filename_sig)
corlor_list = ['green','red','black']
value_list = [[],[],[],[],[],[]]
ymajorLocator   = MultipleLocator(0.05)
ymajorFormatter   = MultipleLocator(0.01)
fig = plt.figure()

plt.ion()
while(1): 
	for i in range (filenum):
		count = 0
		with open(filename_sig[i], 'r') as f:
			#print filenum
			lines = f.readlines()
			for line in lines:
				#print line
				value = [float(s) for s in line.split(",")]
				value_list[2*i].append(count)
				value_list[2*i+1].append(value[0])
				#value_list[2*i+1].append(1)
				#print (2*i+1)
				count = count + 1
	ax=[]
	ax.append(fig.add_subplot(3,1,1))
	ax.append(fig.add_subplot(3,1,2))
	ax.append(fig.add_subplot(3,1,3))
	for j in range(filenum):

		if j == 0:
			ax[j].set_xlabel("count")
			ax[j].set_ylabel("scale")
		if j == 1:
			ax[j].set_xlabel("count")
			ax[j].set_ylabel("weights_fc1")
		if j == 2:
			ax[j].set_ylabel("weights_fc2")
		
		ax[j].scatter(value_list[2*j], value_list[2*j+1], color=corlor_list[j], label='value', linewidth=0.05)
		
	
	plt.pause(0.1)
	# plt.close()
			
