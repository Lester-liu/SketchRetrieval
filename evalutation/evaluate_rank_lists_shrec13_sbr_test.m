%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                     The evaluation code for 
%%% SHREC'12 -- Sketch-Based 3D Shape Retrieval   
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%Input:
%       rank list files (training or testing dataset by changing the 3rd
%       clause)
%       
%       classification files (training or testing dataset, target model dataset) 

%Output:
%       Precision Recall performance                         -- "PR.txt"
%       Other performance metrics (each model and average)   -- "Stats.txt"

%Evaluation measures:
%       NN, 1-Tier, 2-Tier, e_Measure, DCG

%Author: Bo Li
%Email: li.bo.ntu0@gmail.com, libo0002@ntu.edu.sg, b_l58@txstate.edu
%Date: January 24, 2012
%@Texas State University, Texas, US

%Please cite:
% SHREC'13 -- Large Sketch-Based 3D Shape Retrieval, 3DOR'13, 2013

%Note: please note the comments started with "%Chang..." to adjust to your
%cases

clear all;
clc

%Change this clause accordingly: training, testing or complete
%[sketchclass, Ns]=read_classification_file_shrec13('SHREC13_SBR_Sketch_Train.cla');
[sketchclass,Ns]=read_classification_file_shrec13('SHREC13_SBR_Sketch_Test.cla'); 
%[sketchclass, Ns]=read_classification_file_shrec13('SHREC13_SBR_Sketch.cla');

[modelclass,N]=read_classification_file_shrec13('SHREC13_SBR_Model.cla');

number_of_queries=size(sketchclass,1);
number_of_target=size(modelclass,1);
number_of_classes=size(N,1);
  
for qqq=1:number_of_queries 
    for ccc=1:number_of_classes
        if (strcmp(sketchclass{qqq,2},N{ccc,1})==1)
            C(qqq,1)=N{ccc,2};
            break;
        end
    end
end
   
%Change the folder that contains the rank list files accordingly
RANK_DIR='F:\Projects\SHREC 2013\Evaluation\SHREC2013_Sketch_Evaluation\RetrievalLists\';

%output example retrieval lists for testing.
%for i=1:number_of_queries
%    retrievallistfilename=[RANK_DIR sketchclass{i,1}];
%    fp=fopen(retrievallistfilename,'w');
%    fprintf(fp,'%s\n',sketchclass{i,1});
%    for j=1:number_of_target
%        fprintf(fp,'m%s %f\n',modelclass{j,1},j);
%    end
%    fclose(fp);
%end

P_points=zeros(number_of_queries,max(C));
Av_Precision=zeros(1,number_of_queries); 
NN=zeros(1,number_of_queries);
FT=zeros(1,number_of_queries);
ST=zeros(1,number_of_queries);
dcg=zeros(1,number_of_queries);
E=zeros(1,number_of_queries);
filename=[RANK_DIR 'Stats_test.txt'];
fid=fopen(filename,'w');
fprintf(fid,'        NN     FT     ST      E       DCG\n');
fclose(fid); 
for qqq=1:number_of_queries    
    query_name=sketchclass{qqq,1};
    filename=[RANK_DIR query_name];
    
    [R,S,q]=read_rank_list_shrec13(filename,number_of_target);

    query_class=sketchclass{qqq,2};
    
    retrieved_class=cell(1,length(R));
    G=zeros(1,number_of_target);
    for r=1:number_of_target
        for i=1:number_of_target
            if (strcmp(R{r}(2:end),modelclass{i,1})==1)                        
                n=i;
                break;
            end
        end
        retrieved_class{1,r}=modelclass{n,2};
        if (strcmp(query_class,retrieved_class{1,r})==1)
            G(r)=1;
        end
    end;
    
    G_sum=cumsum(G); 
    
    R_points=zeros(1,C(qqq));
    for rec=1:C(qqq)
        R_points(rec)=find((G_sum==rec),1);
    end;
    
    P_points(qqq,1:C(qqq))=G_sum(R_points)./R_points;
    Av_Precision(qqq)=mean(P_points(qqq,:));
    
    NN(qqq)=G(1);
    FT(qqq)=G_sum(C(qqq))/C(qqq);
    ST(qqq)=G_sum(2*C(qqq))/C(qqq);
    P_32=G_sum(32)/32;
    R_32=G_sum(32)/C(qqq);
    
    if (P_32==0)&&(R_32==0);
        E(qqq)=0;
    else
        E(qqq)=2*P_32*R_32/(P_32+R_32);
    end;
    
    NORM_VALUE=1+sum(1./log2([2:C(qqq)]));
    dcg_i=(1./log2([2:length(R)])).*G(2:end);
    dcg_i=[G(1);dcg_i(:)];
    dcg(qqq)=sum(dcg_i)/NORM_VALUE;
    
    filename=[RANK_DIR 'Stats_test.txt'];
    fid=fopen(filename,'a');
    fprintf(fid,'No.%d: %2.3f\t %2.3f\t %2.3f\t %2.3f\t %2.3f\n',qqq, NN(qqq),FT(qqq),ST(qqq),E(qqq),dcg(qqq));
    fclose(fid);    
end;

NN_av=mean(NN)
FT_av=mean(FT)
ST_av=mean(ST)
dcg_av=mean(dcg)
E_av=mean(E)
Mean_Av_Precision=mean(Av_Precision) 
filename=[RANK_DIR 'Stats_test.txt'];
fid=fopen(filename,'a');
fprintf(fid,'NN       FT       ST       E       DCG \n');
fprintf(fid,'%2.3f\t %2.3f\t %2.3f\t %2.3f\t %2.3f\n',NN_av,FT_av,ST_av,E_av,dcg_av);
fclose(fid);

filename=[RANK_DIR 'PR_test.txt'];
fid=fopen(filename,'w');
fprintf(fid,'Recall      Precision\n');
fclose(fid); 
Precision=calcAvgPerf(P_points, C, number_of_queries, filename);
Recall=[1:20]/20;

plot(Recall,Precision);
hold on;
plot(Recall,Precision,'*');
axis([0 1 0 1])
hold off;

xlabel('Precision'),ylabel('Recall')
