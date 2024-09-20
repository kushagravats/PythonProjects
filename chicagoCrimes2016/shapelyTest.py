import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from matplotlib.colors import Normalize
import neighborhoodize

hood_map = neighborhoodize.NeighborhoodMap(neighborhoodize.zillow.ILLINOIS)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)

mymap = Basemap(llcrnrlon=-87.9,
                llcrnrlat=41.3,
                urcrnrlon=-86.91,
                urcrnrlat=42.7,
                projection='lcc', lat_1=32, lat_2=45, lon_0=-95,
                resolution='h')

mymap.drawmapboundary(fill_color='#46bcec')
mymap.fillcontinents(color='#f2f2f2', lake_color='#46bcec')
mymap.drawcoastlines()

for hood in hood_map.neighborhoods:
    lon, lat = hood.polygon.exterior.coords.xy
    x, y = mymap(lon, lat)
    mymap.plot(x, y, '-k', alpha=0.5)
  
crime_data = [(41.8, -87.6), (42.0, -87.7), (41.9, -87.5)] 
for crime in crime_data:
    x, y = mymap(crime[1], crime[0])
    mymap.plot(x, y, 'ro', markersize=5, alpha=0.7)

plt.show()
