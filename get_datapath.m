function path = get_datapath(Dateset, Splitmode, ImgFtSource, lambda1, lambda2, istrain) 
    path=[];
    if(strcmp(Dateset, 'CUBird'))

        datapath = './dataset/CUB2011';
        repath   = './CUBirdResult'; 

        text_feat_path = [datapath, '/11083D_TFIDF.mat'];
        img_feat_path  = [datapath, '/cnn_feat_7part_DET_ReLU.mat'];
        img_label_path = [datapath, '/image_class_labels.mat'];
        disp(['Dataset: CUB2011   ', Splitmode, '  ', ImgFtSource]);
        if(istrain)
            fprintf('Parameter  %d  %d\n', lambda1, lambda2); 
        end
        if(strcmp(Splitmode, 'Easy'))
            data_split_path = [datapath, '/train_test_split_easy.mat'];
            if(strcmp(ImgFtSource, 'DET'))
                img_feat_path  = [datapath, '/cnn_feat_7part_DET_ReLU.mat'];
            elseif(strcmp(ImgFtSource, 'ATN'))
                img_feat_path  = [datapath, '/cnn_feat_7part_ATN_ReLU.mat'];
            end
        elseif(strcmp(Splitmode, 'Hard'))
            data_split_path = [datapath, '/train_test_split_hard.mat'];
            
            if(strcmp(ImgFtSource, 'DET'))
                img_feat_path  = [datapath, '/cnn_feat_7part_DET_ReLU_hard.mat'];
            else
                error('This setting is not available now.')
            end
        end

    elseif(strcmp(Dateset, 'NABird'))

        datapath = './dataset/NABird';
        repath = './NABirdResult'; 

        text_feat_path = [datapath, '/13585D_TFIDF_NABird.mat'];
        img_feat_path  = [datapath, '/cnn_feat_6part_DET_NABird.mat'];
        img_label_path = [datapath, '/image_class_labels_NABird.mat'];

        disp(['Dataset: NABird   ',  Splitmode, '  ', ImgFtSource]);
        if(istrain)
            fprintf('Parameter  %d  %d\n', lambda1, lambda2); 
        end
        
        if(strcmp(Splitmode, 'Easy'))
            data_split_path = [datapath, '/train_test_split_NABird_easy.mat'];
            if(strcmp(ImgFtSource, 'DET'))
                img_feat_path  = [datapath, '/cnn_feat_6part_DET_NABird.mat'];
            elseif(strcmp(ImgFtSource, 'ATN'))
                img_feat_path  = [datapath, '/cnn_feat_6part_ATN_NABird.mat'];
            end
        elseif(strcmp(Splitmode, 'Hard'))
            data_split_path = [datapath, '/train_test_split_NABird_hard.mat'];
            
            if(strcmp(ImgFtSource, 'DET'))
                img_feat_path  = [datapath, '/cnn_feat_6part_DET_NABird_hard.mat'];
            else
                error('This setting is not available now.')
            end
        end
    else
        error('unsupported Dateset, You need prepare it first.\n');
    end
    
    if(istrain)
        if(lambda1 ~=0 ) param1 = log10(lambda1); else param1 =0; end
        if(lambda2 ~=0 ) param2 = log10(lambda2); else param2 =0; end
        %%%% prepare output path
        if(~exist(repath, 'dir')) mkdir(repath); end
        repath = sprintf('%s/%s_%s_Param_%s_%s_%s', repath, Dateset, Splitmode, num2str(param1), num2str(param2), ImgFtSource);
        if(~exist(repath, 'dir')) mkdir(repath); end
        disp(['Result stored in:', repath]); 
        path.repath = repath; 
    end
    
    path.text_feat_path = text_feat_path;
    path.img_feat_path  = img_feat_path;
    path.img_label_path = img_label_path;
    path.data_split_path = data_split_path; 
    
    
end 