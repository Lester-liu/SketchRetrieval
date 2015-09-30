%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% read a retrieval list for a query sketch (model names, distances).
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [R,S,q]=read_rank_list(filename,number_of_target)

fp=fopen(filename,'r');
q{1}=fscanf(fp,'%s',1);
for i=1:number_of_target
    R{i,:}=fscanf(fp,'%s',1);
    S{i,:}=fscanf(fp,'%f',1);
end
fclose(fp);

    
    