import os
import shutil
import time
from datetime import datetime

def cleanUp(path):
    #Get current timestamp to used as name
    timeObj = datetime.now().time()
    timeStamp = timeObj.strftime("%Y_%B_%d")
    timeStr = "ML_batch_" + timeStamp
    
    #Creating output paths
    output_folder = os.path.join(path, timeStr)
    model_folder = os.path.join(output_folder, "Pickled_models")
    graph_folder = os.path.join(output_folder, "Graphs")
    
    #Create folders
    os.mkdir(output_folder)
    os.mkdir(model_folder)
    os.mkdir(graph_folder)
    
    #Move pickled folders
    modelList = os.listdir(os.path.join('Pickled_models'))
    for f in modelList:
        shutil.move(os.path.join('Pickled_models', f), model_folder)
    
    #Move graphs
    graphList= [file for file in os.listdir() if file.endswith('.png')]
    for f in graphList:
        shutil.move(f, graph_folder)
        
    #Move accuracy result file
    accuracyFiles= [file for file in os.listdir() if file.startswith('AccuracyResults')]
    for f in accuracyFiles:
        shutil.move(f, output_folder)

# if __name__ == '__main__':
    # cleanUp('')