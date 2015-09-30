1. Classification files
   Target model dataset    --- SHREC13_SBR_Model.cla 
   Training sketch dataset --- SHREC13_SBR_Sketch_Train.cla
   Testing sketch dataset  --- SHREC13_SBR_Sketch_Test.cla
2. Code
   evaluate_rank_lists_shrec13_sbr.m  --- main file to evaluate on training/testing/complete datasets
   read_classification_file_shrec13.m --- read a sketch/model classification file (models' name and class, class information)
   read_rank_list_shrec13.m           --- read a retrieval list for a query sketch (model names, distances)
   calcAvgPerf.m                      --- calculate average precision-recall based on averaging on all the models
   interpolatePerf.m                  --- compute the precision value by bilinearly interpolation on the neighboring precision values 
      
