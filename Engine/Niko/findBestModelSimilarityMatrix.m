function [rs,ps,bestModelI]=findBestModelSimilarityMatrix(pats,modelSimMats_utv)
% tests whether the model similarity matrix modelSimMats_utv is significantly
% rank-correlated to the similarity matrix of the activity patterns
% obtained when fMRI time-space data matrix Y is modeled with design matrix
% X.
%
% ARGUMENTS
% modelSimMats_utv
%       the model similarity matrices (nModels many rows of vectorized
%       upper triangular similarity matrices)




%% similarity matrix from the data
%simMat_utv=pdist(pats,'euclidean');
simMat_utv=pdist(pats,'correlation');


%% rank correlation between actual and each model similarity matrix
[rs, ps]=corr(simMat_utv',modelSimMats_utv','type','Spearman','rows','pairwise');  % correlation types: Pearson (linear), Spearman (rank), Kendall (rank)
% setting the 'rows' parameter to 'pairwise' makes the function ignore
% nans.


%% determine best-fitting model
[ignore,bestModelI]=max(rs);
