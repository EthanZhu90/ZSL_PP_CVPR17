function [ ] = ZSL_Test(Dateset, Splitmode, ImgFtSource, modelpath)
% example
% ZSL_Test('CUBird', 'Easy', 'DET', 'trained_models/CUBird_Easy_DET.mat')

%%% specify the setting you want to test, and path to the trained model
if(~exist('Dateset', 'var'))    Dateset = 'CUBird';  end % {'CUBird', 'NABird'}
if(~exist('Splitmode', 'var'))  Splitmode = 'Easy';  end % {'Easy', 'Hard'}
% feature extracted based on (1)detected boundingbox or (2)annotation. 
if(~exist('ImgFtSource', 'var')) ImgFtSource = 'DET'; end % {'DET', 'ATN'} 
if(~exist('modelpath', 'var')) 
    if(strcmp(Dateset, 'CUBird')&&strcmp(Splitmode, 'Easy')&&strcmp(ImgFtSource, 'DET'))
        modelpath = 'trained_models/CUBird_Easy_DET.mat';
    elseif(strcmp(Dateset, 'CUBird')&&strcmp(Splitmode, 'Easy')&&strcmp(ImgFtSource, 'ATN'))
        modelpath = 'trained_models/CUBird_Easy_ATN.mat';
    elseif(strcmp(Dateset, 'CUBird')&&strcmp(Splitmode, 'Hard')&&strcmp(ImgFtSource, 'DET'))
        modelpath = 'trained_models/CUBird_Hard_DET.mat';
    elseif(strcmp(Dateset, 'NABird')&&strcmp(Splitmode, 'Easy')&&strcmp(ImgFtSource, 'DET'))
        modelpath = 'trained_models/NABird_Easy_DET.mat';
    elseif(strcmp(Dateset, 'NABird')&&strcmp(Splitmode, 'Hard')&&strcmp(ImgFtSource, 'DET'))
        modelpath = 'trained_models/NABird_Hard_DET.mat';
    else
        error('You need to provide a trained model. ')
    end
end

model = load(modelpath);
path = get_datapath(Dateset, Splitmode, ImgFtSource, 0, 0, false);
fprintf('Model: %s\n', modelpath)

%%%%  prepare the data for testing.
img_feat_dict =  load(path.img_feat_path);
text_feat_dict = load(path.text_feat_path); 
img_label_dict = load(path.img_label_path);
data_split_dict  = load(path.data_split_path); 

label = img_label_dict.imageClassLabels(:, 2);
Data = double(img_feat_dict.cnn_feat');

ctr = data_split_dict.train_cid;
cte = data_split_dict.test_cid;

NumTrnClass = length(unique(ctr));
NumTstClass = length(unique(cte));
fprintf('Load Testing set\n')

NumClass = NumTrnClass + NumTstClass;
nPerClass = zeros(NumClass, 1);
IdPerClass = cell(NumClass, 1);
for idc = 1:NumClass
    
    IdPerClass{idc} = find(label==idc);
    nPerClass(idc) = sum(label==idc);  
end

Xte = [];yte = [];
for idc = cte
    Xc = Data(IdPerClass{idc}, :);
    Xte = [Xte; Xc];
    yte = [yte; idc*ones(size(Xc,1),1)];
end

N_te = length(yte);
y_te = zeros(N_te, 1); 
for n =1:N_te
    y_te(n) = find(cte==yte(n));
end

Z_te  = text_feat_dict.PredicateMatrix(cte, :)';

%%%% Test and display
fprintf('test_acc = %1.4f%%  \n', 100 * (1-get_error(Xte', model.W_x_opt, model.W_z_opt , Z_te, y_te)));

end

function err = get_error(X, W_x, W_z, Z, y)
    pred_score =X' * W_x' * W_z * Z;
    [~, maxIdx] = max(pred_score');
    pred_id = maxIdx';
    GT_id = y;
    err = sum(pred_id ~= GT_id) / length(y);
end
