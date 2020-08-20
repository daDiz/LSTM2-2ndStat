############################################
#  plot heatmap & calculate correlation
#
############################################
import numpy as np
from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import csv
import cPickle
from collections import Counter
import matplotlib.colors as colors
import argparse

parser = argparse.ArgumentParser(description='heatmap')

parser.add_argument('name', type=str, help='method name, e.g. real, LC, LC+LK')

args = parser.parse_args()

def seq2coord(seq, coords):
    c = np.array([coords[x-1][1:] for x in seq])
    return c


coords = np.loadtxt('eq_1997-2018_coords.txt', delimiter=' ', dtype=float)

with open('./beta0_dim64_lr1.0/seq_real.pkl') as file:
    seq_real = cPickle.load(file)

with open('./beta0_dim64_lr1.0/seq_fake.pkl') as file:
    seq_fake_2 = cPickle.load(file)[32:]

with open('./beta0.1_dim64_lr1.0/seq_fake.pkl') as file:
    seq_fake_2nd = cPickle.load(file)[32:]


c_real = seq2coord(seq_real, coords)
c_fake_2 = seq2coord(seq_fake_2, coords)
c_fake_2nd = seq2coord(seq_fake_2nd, coords)


start = 0
end = len(seq_real)
lons_real = c_real[start:end,0]
lats_real = c_real[start:end,1]
lons_fake_2 = c_fake_2[start:end,0]
lats_fake_2 = c_fake_2[start:end,1]
lons_fake_2nd = c_fake_2nd[start:end, 0]
lats_fake_2nd = c_fake_2nd[start:end, 1]

label = args.name
if label == 'real':
    lons = lons_real
    lats = lats_real
elif label == 'LC':
    lons = lons_fake_2
    lats = lats_fake_2
elif label == 'LC+LK':
    lons = lons_fake_2nd
    lats = lats_fake_2nd

#plt.figure(figsize = (8, 8))
m = Basemap(projection='mill', llcrnrlon=-121, llcrnrlat=31,
        urcrnrlon=-113, urcrnrlat=37, resolution = 'l', epsg = 4269)

m.drawcoastlines(linewidth=2.)
m.drawcountries(linewidth=2.)
m.drawstates(linewidth=2.)

parallels = np.arange(31,37,2)
m.drawparallels(parallels,labels=[False,True,True,False],fontsize=16,weight='bold')
meridians = np.arange(-121,-113,2)
m.drawmeridians(meridians,labels=[True,False,False,True],fontsize=16,weight='bold')
#######################################################################

# compute appropriate bins to aggregate data
# nx is number of bins in x-axis, i.e. longitude
# ny is number of bins in y-axis, i.e. latitude
nx = 6*5 # 10 degree for longitude bin
ny = 8*5 # 10 degree for latitude bin
db = 0.5

# form the bins
lon_bins = np.linspace(-121-db, -113+db, nx)
lat_bins = np.linspace(31-db, 37+db, ny)

# aggregate the number of earthquakes in each bin, we will only use the density
density, lat_edges, lon_edges = np.histogram2d(lats, lons, [lat_bins, lon_bins], density=False)



# get the mesh for the lat and lon
lon_bins_2d, lat_bins_2d = np.meshgrid(lon_bins, lat_bins)

# convert the bin mesh to map coordinates:
xs, ys = m(lon_bins_2d, lat_bins_2d) # will be plotted using pcolormesh
#print(xs)
#print(ys)
# define custom colormap, white -> red, #E6072A = RGB(0.9,0.03,0.16)
cdict = {'red':  ( (0.0,  1.0,  1.0),
                   (1.0,  0.9,  1.0) ),
         'green':( (0.0,  1.0,  1.0),
                   (1.0,  0.03, 0.0) ),
         'blue': ( (0.0,  1.0,  1.0),
                   (1.0,  0.16, 0.0) ) }
custom_map = LinearSegmentedColormap('custom_map', cdict)
plt.register_cmap(cmap=custom_map)

# Here adding one row and column at the end of the matrix, so that 
# density has same dimension as xs, ys, otherwise, using shading='gouraud'
# will raise error
density = np.hstack((density,np.zeros((density.shape[0],1))))
density = np.vstack((density,np.zeros((density.shape[1]))))


#bounds = np.linspace(0, np.max(density), 1000)
#norm = colors.BoundaryNorm(boundaries=bounds, ncolors=256)
norm = colors.PowerNorm(gamma=1./3.)
#norm = colors.LogNorm(vmin=1, vmax=np.max(density))
pcm = plt.pcolormesh(xs, ys, density, norm=norm, cmap='custom_map', shading='gouraud')
#cbar = plt.colorbar(extend='max', orientation='vertical')

# Plot heatmap with the custom color map
#plt.pcolormesh(xs, ys, density, cmap="custom_map", shading='gouraud')

# Add color bar and 
cbar = plt.colorbar(orientation='vertical', shrink=0.6, aspect=20, fraction=0.1,pad=0.08)
cbar.ax.tick_params(labelsize=16)

cbar.set_label('Number of earthquakes',size=16, weight='bold')
#cbar = plt.colorbar(orientation='vertical', shrink=0.625, aspect=20, fraction=0.1,pad=0.08)
#cbar.set_label('Earthquake Probability',size=10)
# Plot blue scatter plot of epicenters above the heatmap:    
x,y = m(lons, lats)
#m.plot(x, y, 'o', markersize=7,zorder=6, markerfacecolor='#424FA4',markeredgecolor="none", alpha=0.5)
m.scatter(x, y, marker = 'o', color='b', alpha=0.1, label=label)
# make image bigger:
plt.gcf().set_size_inches(12,12)
plt.legend(loc='upper right', prop={'size':16, 'weight':'bold'})
#plt.show()
plt.savefig('./eq_heatmap_%s.pdf' % label, bbox_inches = 'tight', pad_inches = 0)


############################
# calculate correlation
#
############################

# compute appropriate bins to aggregate data
# nx is number of bins in x-axis, i.e. longitude
# ny is number of bins in y-axis, i.e. latitude
nx = 6*5 # 10 degree for longitude bin
ny = 8*5 # 10 degree for latitude bin
db = 0.5

# form the bins
lon_bins = np.linspace(-121-db, -113+db, nx)
lat_bins = np.linspace(31-db, 37+db, ny)


# aggregate the number of earthquakes in each bin, we will only use the density
density_real, lat_edges_real, lon_edges_real = np.histogram2d(lats_real, lons_real, [lat_bins, lon_bins], density=False)
density_fake_2, lat_edges_fake_2, lon_edges_fake_2 = np.histogram2d(lats_fake_2, lons_fake_2, [lat_bins, lon_bins], density=False)
density_fake_2nd, lat_edges_fake_2nd, lon_edges_fake_2nd = np.histogram2d(lats_fake_2nd, lons_fake_2nd, [lat_bins, lon_bins], density=False)

corr_2 = np.corrcoef(density_real.reshape(-1), density_fake_2.reshape(-1))
corr_2nd = np.corrcoef(density_real.reshape(-1), density_fake_2nd.reshape(-1))

print('corr LC: %.3f' % (corr_2[0,1]))
print('corr LC+LK: %.3f' % (corr_2nd[0,1]))


