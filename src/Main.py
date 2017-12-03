from src.data_loader.Data_Formatter import Data_Formatter
from src.visual_handles.Graph_Factory import GraphFactory
from multiprocessing import Process
#
df = Data_Formatter('../data/EEG/EEGEyeState.csv').get_df()
gf = GraphFactory('../visuals/')
#
if __name__ == '__main__':
  p1 = Process(target=gf.scatter_plot, args=[df[['F3']],df[['F7']],df[['eyeDetection']], False])
  p1.start()
  p2 = Process(target=gf.scatter_plot, args=[df[['T7']],df[['FC']],df[['eyeDetection']], False])
  p2.start()
  p3 = Process(target=gf.scatter_plot, args=[df[['P7']],df[['F3']],df[['eyeDetection']], False])
  p3.start()
  p4 = Process(target=gf.scatter_plot, args=[df[['O1']],df[['FC']],df[['eyeDetection']], False])
  p4.start()
  p5 = Process(target=gf.scatter_plot, args=[df[['O1']],df[['T7']],df[['eyeDetection']], False])
  p5.start()
  p6 = Process(target=gf.scatter_plot, args=[df[['O2']],df[['F7']],df[['eyeDetection']], False])
  p6.start()
  p7 = Process(target=gf.scatter_plot, args=[df[['P8']],df[['AF3']],df[['eyeDetection']], False])
  p7.start()
  p8 = Process(target=gf.scatter_plot, args=[df[['T8']],df[['FC']],df[['eyeDetection']], False])
  p8.start()
  p9 = Process(target=gf.scatter_plot, args=[df[['T8']],df[['O1']],df[['eyeDetection']], False])
  p9.start()
  p10 = Process(target=gf.scatter_plot, args=[df[['T8']],df[['O2']],df[['eyeDetection']], False])
  p10.start()
  p11 = Process(target=gf.scatter_plot, args=[df[['FC']],df[['AF3']],df[['eyeDetection']], False])
  p11.start()
  p11 = Process(target=gf.scatter_plot, args=[df[['FC']],df[['F3']],df[['eyeDetection']], False])
  p11.start()
  p12 = Process(target=gf.scatter_plot, args=[df[['FC']], df[['P8']], df[['eyeDetection']], False])
  p12.start()
  p13 = Process(target=gf.scatter_plot, args=[df[['F4']], df[['FC']], df[['eyeDetection']], False])
  p13.start()
  p14 = Process(target=gf.scatter_plot, args=[df[['F4']], df[['O1']], df[['eyeDetection']], False])
  p14.start()
  p15 = Process(target=gf.scatter_plot, args=[df[['F4']], df[['O2']], df[['eyeDetection']], False])
  p15.start()
  p16 = Process(target=gf.scatter_plot, args=[df[['F8']], df[['AF3']], df[['eyeDetection']], False])
  p16.start()
  p17 = Process(target=gf.scatter_plot, args=[df[['F8']], df[['F3']], df[['eyeDetection']], False])
  p17.start()
  p18 = Process(target=gf.scatter_plot, args=[df[['F8']], df[['P8']], df[['eyeDetection']], False])
  p18.start()
  p19 = Process(target=gf.scatter_plot, args=[df[['F8']], df[['FC']], df[['eyeDetection']], False])
  p19.start()
  p20 = Process(target=gf.scatter_plot, args=[df[['AF4']], df[['F3']], df[['eyeDetection']], False])
  p20.start()
  p21 = Process(target=gf.scatter_plot, args=[df[['AF4']], df[['P7']], df[['eyeDetection']], False])
  p21.start()
  p1.join()
  p2.join()
  p3.join()
  p4.join()
  p5.join()
  p6.join()
  p7.join()
  p8.join()
  p9.join()
  p10.join()
  p11.join()
  p12.join()
  p13.join()
  p14.join()
  p15.join()
  p16.join()
  p17.join()
  p18.join()
  p19.join()
  p20.join()
  p21.join()
