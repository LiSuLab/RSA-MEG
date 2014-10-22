%  modelRDMs is a user-editable function which specifies the models which
%  brain-region RDMs should be compared to, and which specifies which kinds of
%  analysis should be performed.
%
%  Models should be stored in the "Models" struct as a single field labeled
%  with the model's name (use underscores in stead of spaces).
%  
%  Cai Wingfield 11-2009

function Models = modelRDMs_demo()

Models.all_separate = kron([
			0 1 1 1
			1 0 1 1
			1 1 0 1
			1 1 1 0], ones(16,16));
Models.main_clusters = kron([
			0 0 1 1
			0 0 1 1
			1 1 0 0
			1 1 0 0], ones(16,16));
Models.first = kron([
			0 .3 1 1
			.3 0 1 1
			1 1 0 .1
			1 1 .1 0], ones(16,16));
Models.second = kron([
			0 .1 1 1
			.1 0 1 1
			1 1 0 .3
			1 1 .3 0], ones(16,16));
Models.bad_prototype = [kron([0 .5; .5 0], ones(16,16)) ones(32,32); ones(32,32), 1-eye(32,32)];
Models.random = squareform(pdist(rand(64,64)));