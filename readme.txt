####################################################################################################
#
# Learning Network Event Sequences Using Long Short-term Memory and Second-order Statistic Loss
#
####################################################################################################

Requirements:

Python 2.7; GPU (tensorflow-gpu) is required. For details, see requirements.txt. 


Folder layout:

code
  |-- datasets
	|-- earthquake
	|-- email
	|-- twitter
	|-- rand-1
	|-- rand-2
	|-- grid
  |-- grid_search (perform hyper-parameter tuning)
	|-- data
	|-- cst
	|-- checkpoint
  |-- train (training, event prediction & sequence generation)
	|-- data
	|-- cst
	|-- checkpoint
  |-- results
	|-- diffusion_snapshots (diffusion on a grid)
		|-- gen_images
		|-- snapshots
	|-- jaccard_index (diffusion Jaccard index)
	|-- plot_heatmap (earthquake heatmap)
	|-- plot_cst (plot loss)


Install:

## navigate to the code diretory
cd ./code

## create virtual environment
virtualenv venv

## activate virtual environment
. venv/bin/activate

## install required packages
pip install -r requirements.txt


Run:

1. grid_search 
	(cd ./grid_search)
	1.1 preprocess
		python preprocess.py name
			inputs: 
				name -- dataset name, select from earthquake, email, twitter, rand-1, rand-2

			e.g. python preprocess.py earthquake
				 
	1.2 do grid search
		./gs.sh name > gs.log
		or
		nohup ./gs.sh name > gs.log &

			inputs: 
				name -- dataset name, selet from earthquake, email, twitter, rand-1, rand-2

	1.3 display the optimal hyper-parameters and the corresponding hit rates
		python find_best.py

2. train
	(cd ./train)
	2.1 preprocess
		python preprocess.py name
			inputs:
				name -- dataset name, select from earthquake, email, twitter, rand-1, rand-2
			e.g. python preprocess.py earthquake

	2.2 train
		2.2.1 activate virtual env
			. ../venv/bin/activate

		2.2.2 start training
			nohup python train.py beta dim lr name > train.log &

			inputs: 
                		beta -- regularizer for 2nd order statistic constraint
                		dim -- hidden dimension size
                		lr -- initial learning rate
                		name -- dataset name, select from earthquake, email, twitter, rand-1, rand-2

		2.2.3 deactivate virtual env
			deactivate
	
	2.3 generate sequence
		2.3.1 activate virtual env
			. ../venv/bin/activate

		2.3.2 generate sequence
			nohup python gen_seq.py beta dim name > gen_seq.log &

			inputs: 
                		beta -- regularizer for 2nd order statistic constraint
                		dim -- hidden dimension size
                		name -- dataset name, select from earthquake, email, twitter, rand-1, rand-2
			outputs: 
                		seq_real.pkl -- real sequence
				seq_fake.pkl -- fake sequence

		2.3.3 deactivate virtual env
			deactivate

	2.4 predict next event
		2.4.1 activate virtual env
			. ../venv/bin/activate

		2.4.2 predict next event 
			nohup python predict.py beta dim name > predict.log &

			inputs:
                		beta -- regularizer for 2nd order statistic constraint
                		dim -- hidden dimension size
                		name -- dataset name, select from earthquake, email, twitter, rand-1, rand-2
			outputs: 
                		dist_fake.pkl -- probability distribution for the next event
		
		2.4.3 deactivate virtual env
			deactivate

	2.5 calculate hit rates
		python hit.py > hit_rate.txt

		Note: the real sequence incudes the first 32 events while the fake one dosn't. 


3. results/diffusion_snapshots
	To generate snapshots: (navigate to gen_images/)

		python gen_gif.py name
			inputs:
				name -- select from real, LC, LC_LK, hp, rnnpp
			outputs:
				snapshots in ./images

	Note: this is for the grid dataset.


4. results/jaccard_index
	To calculate Jaccard Index:

		python get_index.py > js_index.txt

5. results/plot_heatmap
	To generate earthquake heatmaps:

		python plot_heatmap.py name
			inputs:
				name -- select from real, LC, LC+LK

6. results/plot_cst
	To plot loss:
		
		python plot_cst.py
  
