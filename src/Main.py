from src.data_loader.Data_Formatter import Data_Formatter
from src.visual_handles.Graph_Factory import GraphFactory
from multiprocessing import Pool
#
df = Data_Formatter('../data/EEG/EEGEyeState.csv').get_df()
gf = GraphFactory('../visuals/')
pool = Pool(2)
#
# p1 = pool.apply_async(gf.scatter_plot, [df[['AF3']],df[['F7']],df[['eyeDetection']], False])
# output1 = p1.get(timeout=10)
pool.map(gf.scatter_plot, [df[['AF3']],df[['F7']],df[['eyeDetection']], False])


#gf.scatter_plot(df[['AF3']],df[['F7']],df[['eyeDetection']], False)