function glm_MEGSearchlight_source(subjectNumber, Models, userOptions)

returnHere = pwd; % We'll come back here later
tempBetas = userOptions.betaCorrespondence;
subject = userOptions.subjectNames{subjectNumber};
nSubjects = userOptions.nSubjects;

promptOptions.functionCaller = 'MEGSearchlight_source';
promptOptions.defaultResponse = 'S';
overwriteFlag = overwritePrompt(userOptions, promptOptions);

if overwriteFlag
    
    fprintf('Shining RSA searchlights...\n');
    gotoDir(userOptions.rootPath, 'Maps');
    
    tic;%1
    
    fprintf(['\tSearching in the source meshes of subject ' num2str(subjectNumber) ' of ' num2str(nSubjects) ':']);
    
    % Run searchlight on both halves of the brain
    sourceMeshes = MEGDataPreparation_source(subjectNumber, tempBetas, userOptions);
    
    for chirality = 1:2
        switch chirality
            case 1
                chi = 'L';
            case 2
                chi = 'R';
        end%switch:chirality
        
        % Get masks
        
        % update IZ 03/12
        % assuming every analysis works as a mask, overlaying masks to create a single mask
        nMasks = numel(fieldnames(userOptions.indexMasks));
        indexMasks = userOptions.indexMasks;
        maskNames = fieldnames(userOptions.indexMasks);
        maskIndices=[];
        for mask = 1:nMasks
            thisMaskName = maskNames{mask};
            if strfind(thisMaskName, [lower(chi), 'h'])
                maskIndices = union(maskIndices, indexMasks.(thisMaskName).maskIndices);
                maskIndices = sort(maskIndices(maskIndices <= userOptions.nVertices));
                userOptions.maskIndices.(chi) = maskIndices;
                userOptions.chi = chi;
            end
        end
        timeIndices = userOptions.dataPointsSearchlightLimits;
        
        % Apply masks
        
        if userOptions.nSessions == 1
            maskedMesh = sourceMeshes.(chi)(:, timeIndices(1):timeIndices(2), :); % (vertices, timePointes, conditions)
        else
            maskedMesh = sourceMeshes.(chi)(:, timeIndices(1):timeIndices(2), :, :); % (vertices, timePointes, conditions, sessions)
        end
        [thisSubjectBs.(chi), thisSubjectDs.(chi)] = glm_searchlightMapping_MEG_source(maskedMesh, Models, userOptions);
        
        %% Saving files
        
        bMetadataStruct = userOptions.STCmetaData;
        dMetadataStruct = userOptions.STCmetaData;
        
        for model_i = 1:nModels
            
            bMetadataStruct.data = thisSubjectBs.(chi)(:,:,model_i);
            
            modelName = spacesToUnderscores(Models(model_i).name);
            
            modelDir = fullfile(userOptions.rootPath, 'Maps', modelName);
            gotoDir(fullfile(userOptions.rootPath, 'Maps'), modelName);
            
            outputBFilename = fullfile(modelDir, [userOptions.analysisName '_bMesh_' modelName '_' subject ]);
            if userOptions.maskingFlag
                outputBFilename = [outputBFilename '_masked'];
            end
            
            mne_write_stc_file1([outputBFilename '-' lower(chi) 'h.stc'], bMetadataStruct);
        end
        
        outputDFilename = fullfile(userOptions.rootPath, 'Maps', modelName,  [userOptions.analysisName '_dMesh_' modelName '_' subject ]);
        if userOptions.maskingFlag
            outputDFilename = [outputDFilename '_masked'];
        end
           
        dMetadataStruct.data = thisSubjectDs.(chi); 
        mne_write_stc_file1([outputPFilename '-' lower(chi) 'h.stc'], dMetadataStruct);
        
        userOptions = rmfield(userOptions, 'maskIndices');
        userOptions = rmfield(userOptions, 'chi');
        clear thisSubjectRs thisSubjectPs pMetadataStruct searchlightRDMs sourceMeshes.(chi) maskedMesh;
        
    end%for:chirality
    
    % Print the elapsed time for this subject
    if chirality == 1
        fprintf('\n\t\t\t\t\t\t\t\t');
    else
        t = toc;%1
        fprintf([': [' num2str(ceil(t)) 's]\n']);
    end%if
    
else
    fprintf('Searchlight already applied, skip....\n');
end

cd(returnHere); % And go back to where you started
