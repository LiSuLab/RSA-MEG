%  modelRDMs is a user-editable function which specifies the models which
%  brain-region RDMs should be compared to, and which specifies which kinds of
%  analysis should be performed.
%
%  Models should be stored in the "Models" struct as a single field labeled
%  with the model's name (use underscores in stead of spaces).
%  
%  Cai Wingfield 11-2009

function Models = modelRDMs()

% Models.largerSim = [
%     0 1 1 1 1 1;
%     1 0 1 1 1 1;
%     1 1 0 1 1 1;
%     1 1 1 0 0 0;
%     1 1 1 0 0 0;
%     1 1 1 0 0 0;
%     
% ];

fprintf('Constracting Model RDMs');
load('true_RDM-6.mat');
true_RDMs = averageRDMs_subjectSession(true_RDMs, 'subject');
Models.Noisy = true_RDMs.RDM;

fprintf('Done.\n');

% fprintf('Done.\n');