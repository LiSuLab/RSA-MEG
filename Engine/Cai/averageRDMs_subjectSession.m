% RDMs = averageRDMs_subjectSession(RDMs)
% RDMs = averageRDMs_subjectSession(RDMs, 'subject')
% RDMs = averageRDMs_subjectSession(RDMs, 'session')
% RDMs = averageRDMs_subjectSession(RDMs, 'subject', 'session')
% RDMs = averageRDMs_subjectSession(RDMs, 'session', 'subject')
%
% Averages a struct of RDMs which is [nMasks, nSubjects, nSessions] based on
% arguments and returns the averaged struct.
% changed to nanmean Update: IZ 07/13

function RDMs = averageRDMs_subjectSession(varargin)

	if nargin == 1 || nargin == 2 || nargin == 3

		RDMs = varargin{1};

		if nargin == 2

			if strcmpi(varargin{2}, 'subject')
				RDMs = aSub(RDMs);
			elseif strcmpi(varargin{2}, 'session')
				RDMs = aSes(RDMs);
			else
				wrongStringsError();
			end%if

		elseif nargin == 3

			if strcmpi(varargin{2}, 'subject')
				RDMs = aSub(RDMs);
			elseif strcmpi(varargin{2}, 'session')
				RDMs = aSes(RDMs);
			else
				wrongStringsError();
			end%if

			if strcmpi(varargin{3}, 'subject') && ~strcmpi(varargin{2}, 'subject')
				RDMs = aSub(RDMs);
			elseif strcmpi(varargin{3}, 'session') && ~strcmpi(varargin{2}, 'session')
				RDMs = aSes(RDMs);
			else
				wrongStringsError();
			end%if

		end%if

	else

		wrongNargsError();

	end%if

end%function

%% Subfunctions %%

function wrongStingsError
	error('Please put ''subject'' or ''session'' in for the string arguments.');
end%function

function wrongNargsError
	error('Only accepts 1, 2 or 3 arguments.');
end%function

function aveRDMs = aSub(RDMs)
% 	nMa = size(RDMs, 1);
% 	nSu = size(RDMs, 2);
% 	nSe = size(RDMs, 3);
% 	for ma = 1:nMa
% 		for se = 1:nSe
% 			for su = 1:nSu
% 				if su == 1
% 					suSum = RDMs(ma, su, se).RDM;
% 				else
% 					suSum = suSum + RDMs(ma, su, se).RDM;
% 				end%if:su==1
% 			end%for:su
% 			aveRDMs(ma, 1, se).RDM = suSum ./ nSu;
% 			aveRDMs(ma, 1, se).color = RDMs(ma, 1, se).color;
% 			oldName = RDMs(ma, 1, se).name;
% 			bothBits = findstr(oldName, ' | ');
% 			if numel(bothBits) == 2
% 				firstBit = bothBits(1);
% 				secondBit = bothBits(2);
% 				newName = [oldName(1:firstBit-1) oldName(secondBit:length(oldName))];
% 			elseif numel(bothBits) == 1
% 				firstBit = bothBits(1);
% 				newName = oldName(1:firstBit-1);
% 			else
% 				newName = oldName; %in case naming conventions weren't observed
% 			end%if
% 			aveRDMs(ma, 1, se).name = newName;
% 		end%for:se
% 	end%for:ma
nMasks = size(RDMs, 1);
nSubjects = size(RDMs, 2);
nSessions = size(RDMs, 3);

for mask =1:nMasks
    for session = 1:nSessions
        temp = [];
        for subject = 1:nSubjects
            temp = [temp; vectorizeRDMs(RDMs(mask, subject, session).RDM)];
        end
        aveRDMs(mask,1,session).RDM = squareform(nanmean(temp,1));
        aveRDMs(mask, 1, session).color = RDMs(mask, 1, session).color;
        oldName = RDMs(mask, 1, session).name;
        bothBits = findstr(oldName, ' | ');
        if numel(bothBits) == 2
            firstBit = bothBits(1);
            secondBit = bothBits(2);
            newName = [oldName(1:firstBit-1) oldName(secondBit:length(oldName))];
        elseif numel(bothBits) == 1
            firstBit = bothBits(1);
            newName = oldName(1:firstBit-1);
        else
            newName = oldName; %in case naming conventions weren't observed
        end%if
        aveRDMs(mask, 1, session).name = newName;
    end%for:sesion
end%for:mask

end%function

function aveRDMs = aSes(RDMs)
% 	nMa = size(RDMs, 1);
% 	nSu = size(RDMs, 2);
% 	nSe = size(RDMs, 3);
% 	for ma = 1:nMa
% 		for su = 1:nSu
% 			for se = 1:nSe
% 				if se == 1
% 					seSum = RDMs(ma, su, se).RDM;
% 				else
% 					seSum = seSum + RDMs(ma, su, se).RDM;
% 				end%if:se==1
% 			end%for:se
% 			aveRDMs(ma, su, 1).RDM = seSum ./ nSe;
% 			aveRDMs(ma, su, 1).color = RDMs(ma, 1, se).color;
% 			oldName = RDMs(ma, su, 1).name;
% 			bothBits = findstr(oldName, ' | ');
% 			if numel(bothBits) == 2
% 				firstBit = bothBits(1);
% 				secondBit = bothBits(2);
% 				newName = oldName(1:secondBit-1);
% 			elseif numel(bothBits) == 1
% 				firstBit = bothBits(1);
% 				newName = oldName(1:firstBit-1);
% 			else
% 				newName = oldName;% in case naming conventions weren't observed
% 			end%if
% 			aveRDMs(ma,su, 1).name = newName;
% 		end%for:su
% 	end%for:ma
    
nMasks = size(RDMs, 1);
nSubjects = size(RDMs, 2);
nSessions = size(RDMs, 3);

for mask =1:nMasks
    for subject = 1:nSubjects
        temp = [];
        for session = 1:nSessions
            temp = [temp; vectorizeRDMs(RDMs(mask, subject, session).RDM)];
        end
        aveRDMs(mask,subject,1).RDM = squareform(nanmean(temp,1));
        aveRDMs(mask,subject,1).color = RDMs(mask,subject,1).color;
        oldName = RDMs(mask,subject,1).name;
        bothBits = findstr(oldName, ' | ');
        if numel(bothBits) == 2
            firstBit = bothBits(1);
            secondBit = bothBits(2);
            newName = [oldName(1:firstBit-1) oldName(secondBit:length(oldName))];
        elseif numel(bothBits) == 1
            firstBit = bothBits(1);
            newName = oldName(1:firstBit-1);
        else
            newName = oldName; %in case naming conventions weren't observed
        end%if
        aveRDMs(mask,subject,1).name = newName;
    end%for:sesion
end%for:mask
end%function