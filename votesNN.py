# Take vote of classifiers
import numpy as np 
import pandas as pd 

if __name__ == '__main__':
	hldr = []
	submission = pd.read_csv('NN.prelu.config.0.csv')
	hldr.append(submission["Response"])

	submission = pd.read_csv('NN.prelu.config.1.csv')
	hldr.append(submission["Response"])

	submission = pd.read_csv('NN.prelu.config.2.csv')
	hldr.append(submission["Response"])

	submission = pd.read_csv('NN.prelu.config.3.csv')
	hldr.append(submission["Response"])

	submission = pd.read_csv('NN.prelu.config.4.csv')
	hldr.append(submission["Response"])

	response_vote = np.mean(np.array(hldr),axis=0)
	response_vote = np.round(response_vote).astype(int)

	submission = pd.read_csv('../sample_submission.csv')
	submission['Response'] = response_vote
	submission.to_csv('NN.relu.votes.csv', index=False)