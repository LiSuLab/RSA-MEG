%  betaCorrespondence.m is a simple nullary function which should return three things:
%  	preBeta:	a string which is at the start of each file containing a beta image
%  	betas:	a struct indexed by (session, condition) containing a sting unique to each beta image
%  	postBeta:	a string which is at the end of each file containing a beta image, not containing the file .suffix
%
%	use "[[subjectName]]" as a placeholder for the subject's name as found in userOptions.subjectNames if necessary
%  
%  Cai Wingfield 1-2010, updated by Li Su 3-2012 to read beta names from a
%  text document, in order to handle large number of conditions.

function betas = betaCorrespondence()

readFromFile = false;
preBeta = '[[subjectName]]/signal_[[subjectName]]-';
%preBeta = '/[[subjectName]]/';
%% == Stem + Affix model == %%

betas(1,1).identifier = 'data1';
betas(1,2).identifier = 'data2';
betas(1,3).identifier = 'data3';
betas(1,4).identifier = 'data4';
betas(1,5).identifier = 'data5';
betas(1,6).identifier = 'data6';
betas(1,7).identifier = 'data7';
betas(1,8).identifier = 'data8';
betas(1,9).identifier = 'data9';
betas(1,10).identifier = 'data10';
betas(1,11).identifier = 'data11';
betas(1,12).identifier = 'data12';
betas(1,13).identifier = 'data13';
betas(1,14).identifier = 'data14';
betas(1,15).identifier = 'data15';
betas(1,16).identifier = 'data16';
betas(1,17).identifier = 'data17';
betas(1,18).identifier = 'data18';
betas(1,19).identifier = 'data19';
betas(1,20).identifier = 'data20';
% betas(1,1).identifier = 'average_length_words_3';
% betas(1,2).identifier = 'average_length_words_4';
% betas(1,3).identifier = 'average_length_words_5';
% betas(1,4).identifier = 'average_length_words_6';
% betas(1,5).identifier = 'average_length_words_7';
% betas(1,6).identifier = 'average_length_words_8';

if readFromFile
    betaFilePath = '/imaging/at03/NKG_Data_Sets/LexproMEG/scripts/Stimuli-Lexpro-MEG-Single-col.txt';
    betaNames = textread(betaFilePath,'%s');
    postBeta = '-[[LR]]h.stc';

    session = 1;
    for i = 1:size(betaNames,1)
        betas(session,i).identifier = [preBeta betaNames{i} postBeta];
    end
else
    postBeta = '.fif';
    for session = 1:size(betas,1)
        for condition = 1:size(betas,2)
            betas(session,condition).identifier = [preBeta betas(session,condition).identifier postBeta];
        end%for
    end%for
end