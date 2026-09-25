function report=calibrate_group(n_rep,n_perm)
%CALIBRATE_GROUP Empirical FWER under predeclared independent Gaussian null.
% Standalone long run: addpath('work/rsa-meeg'); addpath('work/rsa-meeg/tests');
% calibrate_group() uses 1000 datasets, 12 subjects, six connected tests,
% two-sided alpha=.05, 199 Monte Carlo sign flips for each of three methods.
% This is diagnostic calibration, not a fast unit test or a pass/fail gate.
if nargin<1, n_rep=1000; end
if nargin<2, n_perm=199; end
if ~isscalar(n_rep)||n_rep<1||n_rep~=floor(n_rep)||~isscalar(n_perm)||n_perm<1||n_perm~=floor(n_perm)
    error('rsa_meeg:calibration:Input','Repetitions and permutations must be positive integers.');
end
prior=rng; cleanup=onCleanup(@()rng(prior)); rng(20260925,'twister');
nsubj=12; nnode=6; alpha=.05;
adj=false(nnode); for j=1:nnode-1,adj(j,j+1)=true;adj(j+1,j)=true;end
methods={'max-t','cluster-mass','tfce'};
false_positive=zeros(n_rep,numel(methods));
for rep=1:n_rep
    x=randn(nsubj,nnode);
    for j=1:numel(methods)
        o=rsa_meeg.inference.group(x,'Method',methods{j},'Tail','two-sided',...
            'Adjacency',adj,'ClusterThreshold',2,'Alpha',alpha,...
            'NumPermutations',n_perm,'Seed',rep*10+j);
        false_positive(rep,j)=any(o.significant);
    end
end
rate=mean(false_positive,1); ci=zeros(numel(methods),2);
z=1.95996398454005;
for j=1:numel(methods)
    p=rate(j); den=1+z^2/n_rep;
    center=(p+z^2/(2*n_rep))/den;
    half=z*sqrt(p*(1-p)/n_rep+z^2/(4*n_rep^2))/den;
    ci(j,:)=[center-half,center+half];
end
report=table(methods(:),sum(false_positive,1)',rate',ci(:,1),ci(:,2),...
    'VariableNames',{'method','false_positives','empirical_fwer','wilson_low','wilson_high'});
disp(report);
fprintf('Design: %d independent Gaussian null datasets, %d subjects, %d connected nodes, %d permutations, two-sided alpha %.3f.\n',...
    n_rep,nsubj,nnode,n_perm,alpha);
end
