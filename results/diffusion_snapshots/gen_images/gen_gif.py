from __future__ import absolute_import, division, print_function
import cPickle
import matplotlib.pyplot as plt
import imageio
import networkx as nx
import numpy as np
import sys

## given a sequence of events
## generate image files
## ex: gen_images(seq, "data/soc-karate.txt", 'images/soc-karate')
def gen_images(num_frames, seq, graph_file_name, image_name, pos, hist_length):
	if num_frames > len(seq):
		raise ValueError('too many frames')

	g = nx.read_edgelist(graph_file_name, nodetype=int)

	nodes = g.nodes

	val_map = {}
	init = []
	for i in range(num_frames+hist_length):
		s = seq[i]

		if i < hist_length:
			init.append(s)
			val_map[s] = 2.0
		elif s not in init:
			val_map[s] = 1.0

		values = [val_map.get(node, 0.25) for node in nodes]

		nx.draw(g, pos=pos, cmap=plt.get_cmap('jet'), node_color=values, with_labels=True, alpha=0.5)
		plt.savefig("%s%d.png" % (image_name, i))
		plt.clf()

	return

## generate gif from images
## ex: gen_gif(34, "images/soc-karate", "images/soc-karate.gif", duration=3)
def gen_gif(num_frames, image_name, gif_name, duration=5):
	with imageio.get_writer(gif_name, mode='I', duration=duration) as writer:
		for i in range(num_frames):
			file_name = image_name + str(i) + '.png'
			image = imageio.imread(file_name)
			writer.append_data(image)

	return

## given a sequence, generate a gif
def seq2gif(graph_name, seq_name, image_name, num_frames, pos, hist_length, duration=2, do_image=True):
	with open(seq_name, 'rb') as file:
		seq = cPickle.load(file)

	if do_image:
		gen_images(num_frames, seq, graph_name, image_name, pos, hist_length)

	gen_gif(num_frames, image_name, image_name + ".gif", duration=duration)

	return

## given a sequence of events
## generate image files
## ex: gen_images(seq, "data/soc-karate.txt", 'images/soc-karate')
def gen_images_n(step, num_frames, seq, graph_file_name, image_name, pos, hist_length):
    if num_frames > len(seq):
        raise ValueError('too many frames')

    g = nx.read_edgelist(graph_file_name, nodetype=int)
    #pos = nx.spring_layout(g)

    nodes = g.nodes

    val_map = {}
    init = []
    tot = num_frames + hist_length
    for i in range(tot):
        s = seq[i]

        if i < hist_length:
            init.append(s)
            val_map[s] = 2.0
        elif s not in init:
            val_map[s] = 1.0

        values = [val_map.get(node, 0.25) for node in nodes]
        #print(i)
        if i % step == 0:
            #print(i)
            nx.draw(g, pos=pos, cmap=plt.get_cmap('jet'), node_size=100, node_color=values, with_labels=False, alpha=0.5)
            plt.savefig("%s%d.png" % (image_name, i))
            plt.clf()

    return

if __name__ == '__main__':
    name = sys.argv[1] # select from real, LC, LC_LK, hp, rnnpp

    grid_size = 20
    num_frames = 2000
    step = 100
    duration = 0.1
    k = 2
    hist_length = k - 1
    do_image = True
    do_real_gif = False
    do_fake_gif = False

    if name == 'real':
        do_real_image = True
        do_fake_image = False
    else:
        do_real_image = False
        do_fake_image = True

    graph_name = "grid_%s_%s.txt" % (grid_size, grid_size)
    g = nx.read_edgelist(graph_name, nodetype=int)

    pos = {}
    n = 0
    for i in g.nodes:
        x1 = int(n / grid_size)
        x2 = n % grid_size
        pos[i] = np.array([x1, x2])
        n += 1



    print("generating images ...")
    if do_real_image:
        with open('./real/seq_real.pkl') as file:
            seq = cPickle.load(file)

        image_name = 'images/grid_20_20_real'

        gen_images_n(step, num_frames, seq, graph_name, image_name, pos, hist_length)

    if do_fake_image:
        if name == 'hp':
            with open('%s/seq_fake_mark_mle_basis_exp.txt' % (name)) as file:
                seq = np.array(map(int, file.readline().strip('\n').split()))
        else:
            with open('%s/seq_fake.pkl' % (name)) as file:
                seq = cPickle.load(file)

        image_name = 'images/grid_20_20_%s_fake' % (name)

        gen_images_n(step, num_frames, seq, graph_name, image_name, pos, hist_length)


    # real gif
    if do_real_gif:
        real_seq_name = './real/seq_real.pkl'
        real_image_name = 'images/grid_20_20_real'
        seq2gif(graph_name, real_seq_name, real_image_name, num_frames, pos, hist_length, duration=duration, do_image=do_image)


    # fake gif
    if do_fake_gif:
        fake_seq_name = '%s/seq_fake.pkl' % (name)
        fake_image_name = 'images/grid_20_20_%s_fake' % (name)
        seq2gif(graph_name, fake_seq_name, fake_image_name, num_frames, pos, hist_length, duration=duration, do_image=do_image)




