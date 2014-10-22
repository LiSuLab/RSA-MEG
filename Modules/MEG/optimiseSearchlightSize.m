function optimiseSearchlightSize(path,range,left_or_right)

subjectIDs = { ...
		'0144', ...
        '0146', ...      
		'0147', ...
		'0154', ...	
        '0155', ...
		'0156', ...		
		'0157', ... 
		'0161', ...
        '0162', ...
        '0166', ...
        '0169', ...    
		'0172', ...
        '0173', ...	
		'0175', ...
		'0177', ...		
		'0179', ...
        '0180', ...    
		'0187', ...
        '0190', ...	
		'0192'};

baseline = 20; % milliseconds

distance = 'correlation'; % euclidean mahalanobis correlation
          
% The number of subjects
nSubjects = size(subjectIDs,2);
dataPath = strcat('/imaging/cw03/Affix_MEG/RSA/data/source/tworeps/');
number_of_vertex = 10242;

% conditionSetA = 1:6;
% conditionSetB = 13:18;

conditionSetA = 7:12;
conditionSetB = 19:24;

number_of_conditions = size(conditionSetA,2);

% pre-allocate memory for RDMs
%RDM = struct('RDM',mat2cell(ones(number_of_conditions*number_of_timePoints,number_of_conditions*number_of_vertex*2), ...
%    number_of_conditions*ones(1,number_of_timePoints),number_of_conditions*ones(1,number_of_vertex*2)));

lh_Matrix1 = zeros(number_of_conditions,number_of_vertex,109);
lh_Matrix2 = lh_Matrix1;
signal = zeros(50,200,20,15,15); % retest reliability
noise = signal; % null distribution

for j = 1:50
    disp([num2str(j*2) '% finished']);
    presentation = (rand(1,6)>0.5);
    randSetA = conditionSetA+presentation.*12;
    randSetB = conditionSetB-presentation.*12;
    
    for subject = 1:nSubjects
        %disp(['  Calculating RDMs for subject ' subjectIDs{subject} '...']);
        subjectPath = [dataPath, 'morph_', subjectIDs{subject}, '_'];   

        for condition = 1:number_of_conditions % five conditions
                lh_Path = strcat(subjectPath, num2str(randSetA(condition)), '_ons_MEEG_depth-',left_or_right,'.stc'); % Generate filename
                %rh_Path = cell2mat(strcat(subjectPath, conditionSet1(condition), '_allop_rsa-irp_nodepth_mne-rh.stc')); % Generate filename
                lh_Vol = mne_read_stc_file(lh_Path); % Pull in data, requires MNE in the search path
                %rh_Vol = mne_read_stc_file(rh_Path); % Pull in data, requires MNE in the search path
                lh_Matrix1(condition,:,:) = lh_Vol.data;
                %rh_Matrix1(condition,:,:) = rh_Vol.data;

                lh_Path = strcat(subjectPath, num2str(randSetB(condition)), '_ons_MEEG_depth-',left_or_right,'.stc'); % Generate filename
                %rh_Path = cell2mat(strcat(subjectPath, conditionSet2(condition), '_allop_rsa-irp_nodepth_mne-rh.stc')); % Generate filename
                lh_Vol = mne_read_stc_file(lh_Path); % Pull in data, requires MNE in the search path
                %rh_Vol = mne_read_stc_file(rh_Path); % Pull in data, requires MNE in the search path
                lh_Matrix2(condition,:,:) = lh_Vol.data;
                %rh_Matrix2(condition,:,:) = rh_Vol.data;
        end

        for radius = 4:4:60    
            if radius > 4
                adjPath = ['adjacency_' num2str(radius-4) '_' num2str(number_of_vertex) '.mat'];
                load(adjPath);
            end
            for windowSize = 2:16
                onTime = 40;
                offTime = onTime+windowSize;
                timeWindow = (baseline+onTime:baseline+offTime);
                v = 1;
                for i = range
                    if radius > 4
                        near_by_vertex = adjMatrix(i,:);
                        near_by_vertex = near_by_vertex(~isnan(near_by_vertex));
                        current_Vertex = [i, near_by_vertex];
                    else
                        current_Vertex = i;
                    end

                    searchlight1 = lh_Matrix1(:,current_Vertex,timeWindow); % vertex and time points in seachlight1 and seachlight2 are the same
                    condensedData1 = reshape(searchlight1,size(searchlight1,1),size(searchlight1,2).*size(searchlight1,3));  
                    try
                        RDM1 = pdist(squeeze(condensedData1), distance);
                    catch err
                        if (strcmp(err.identifier, 'stats:pdist:InappropriateDistance'))
                            RDM1 = pdist(squeeze(condensedData1), 'euclidean');
                        end
                    end

                    searchlight2 = lh_Matrix2(:,current_Vertex,timeWindow);
                    condensedData2 = reshape(searchlight2,size(searchlight2,1),size(searchlight2,2).*size(searchlight2,3));  
                    try
                        RDM2 = pdist(squeeze(condensedData2), distance);
                    catch err
                        if (strcmp(err.identifier, 'stats:pdist:InappropriateDistance'))
                            RDM2 = pdist(squeeze(condensedData2), 'euclidean');
                        end
                    end

                    %corrType = 'Spearman';
                    signal(j,v,subject,radius./4,windowSize-1) = corr(RDM1', RDM2');%,'type',corrType);
                    noise(j,v,subject,radius./4,windowSize-1) = corr(RDM1', permuteRDM(RDM2)');%,'type',corrType);
                    v = v+1;
                    %e(radius./4,windowSize./5) = entropy(RDM);
                end
            end
        end
    end
end

save([path,'/SNR_',num2str(max(range)),'-',left_or_right,'.mat'],'signal','noise');