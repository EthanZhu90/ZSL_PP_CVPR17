function [ ] = ZSL_Train(Dateset, Splitmode, ImgFtSource, lambda1, lambda2, GPU_mode)
% example
% ZSL_Train('CUBird', 'Easy', 'DET', 100000, 10000, true)

%gpuDevice(2)
if(~exist('Dateset', 'var'))    Dateset = 'CUBird';  end % {'CUBird', 'NABird'}
if(~exist('Splitmode', 'var'))  Splitmode = 'Easy';  end % {'Easy', 'Hard'}
% feature extracted based on (1)detected boundingbox or (2)annotation. 
if(~exist('ImgFtSource', 'var')) ImgFtSource = 'DET'; end % {'DET', 'ATN'} 
if(~exist('lambda1', 'var'))    lambda1 = 100000;   end
if(~exist('lambda2', 'var'))    lambda2 = 10000;   end
if(~exist('GPU_mode', 'var'))      GPU_mode = true;   end
addpath(genpath('./minFunc_2012'))
%%%%  set to True if continuing to train
continueTrain = false;
if(continueTrain)
    continue_weight_path = 'CUBirdResult/CUBird_Easy_Param_5_4_DET/Weight_opt_250.mat'; 
    startLoop = 251; 
end

path = get_datapath(Dateset, Splitmode, ImgFtSource, lambda1, lambda2, true);

if(GPU_mode)  fprintf('Using GPU_mode to train.\n')
else  fprintf('Using CPU_mode to train.\n')
end

%%%%  prepare the data for training. 
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
fprintf('Load training set\n')

NumClass = NumTrnClass + NumTstClass;
nPerClass = zeros(NumClass, 1);
Id_perClass = cell(NumClass, 1);

for idc = 1:NumClass
    Id_perClass{idc} = find(label==idc);
    nPerClass(idc) = sum(label==idc);
end

Xtr = []; ytr = [];
for idc = ctr
    Xc = Data(Id_perClass{idc}, :);
    Xtr = [Xtr; Xc];
    ytr = [ytr; idc*ones(size(Xc,1),1)];
end

C = NumTrnClass;
N = length(ytr);
Y = zeros(N, C);
y = zeros(N, 1);
for n =1:N
    Y(n, :) = ctr==ytr(n);
    y(n) = find(ctr==ytr(n));
end

X   = Xtr';
Z   = text_feat_dict.PredicateMatrix(ctr, :)';
d_x = size(X, 1);
d_z = size(Z, 1);
% Dimension of features for each part
if(~exist('d_p', 'var')) d_p = 512;  end 
% Dimension of embedding space
if(~exist('m', 'var')) m = NumTrnClass;  end 
if(strcmp(Dateset, 'CUBird')) 
    num_Parts = 7; 
elseif(strcmp(Dateset, 'NABird')) 
    num_Parts = 6; 
end

%%%% Set parameter for training 
MAX_ITER = 2;    %%%%   Number of iterations in a loop 
MAX_LOOP = 300;   %%%%   Number of max loops. 
MAX_FUNCEVL =100; %%%%   

options = [];
options.Method = 'lbfgs';
options.Display = 'full';
options.DerivativeCheck = 'off';
options.maxFunEvals = MAX_FUNCEVL;
options.MaxIter = MAX_ITER;

%%%% Initialize weights 
if(continueTrain)
    load(continue_weight_path);
    W_init_x = W_x_opt;
    W_init_z = W_z_opt;
    disp(['Continue training from:',  continue_weight_path]); 
else
    startLoop = 1; 
    W_init_x = randn(m ,d_x); 
    W_init_z = randn(m ,d_z);
    disp('Start from Random Initialization.')
end

if(GPU_mode)
    %%%% prepare gpu data for iteration:
    X = gpuArray(X); 
    Z = gpuArray(Z); 
    Y = gpuArray(Y);
    %%%% prepare gpu data for iteration: End 
end

ZZ_t = Z * Z';
W_x_opt = W_init_x;
W_z_opt = W_init_z; 

fprintf('train_acc = %1.4f%%  \n',  100 * (1-get_error(X, W_x_opt, W_z_opt , Z, y)));
trainWx_FLAG = false; 

for train_Itn = startLoop : MAX_LOOP
    
    t = clock; 
    if(trainWx_FLAG)
        fprintf('\nITER %d:  Training W_x\n', train_Itn);
    else
        fprintf('\nITER %d:  Training W_z\n', train_Itn);
    end
    

    %%%% compute the D_z and D_xz
    D_xzi = zeros(d_z,d_z, num_Parts);
    W_x_t = W_init_x';
    for i = 1:num_Parts
        W_xz = W_x_t((d_p*(i-1)+1) : d_p*(i),:) * W_init_z; 
        D_xzi(:,:,i) = diag([1 ./ (2*sqrt(sum((W_xz').^2,2) + 0.0001))]); 
    end
    
    if(GPU_mode)
        %%%% prepare gpu data inside iteration:
        D_xzi_cell = cell(num_Parts, 1); 
        for i = 1:num_Parts
            D_xzi_cell{i} = gpuArray(sparse(D_xzi(:,:,i))); 
        end
        if(trainWx_FLAG)
            W_init_z = gpuArray(W_init_z); 
        else
            W_init_x = gpuArray(W_init_x);
        end
    else
        D_xzi_cell = cell(num_Parts, 1); 
        for i = 1:num_Parts
            D_xzi_cell{i} = sparse(D_xzi(:,:,i)); 
        end
    end
    
    fprintf('Start training using L-BFGS ......\n')
    if(trainWx_FLAG)
        W_x_opt = minFunc(@ZSL_ObjFunc_Wx, reshape(W_init_x,[m*d_x, 1]), options, num_Parts, m, d_x, W_init_z, ...
            X, Z, Y, ZZ_t, D_xzi_cell, lambda1, lambda2, GPU_mode);
    
        W_x_opt = reshape(W_x_opt, [m, d_x]);
        W_z_opt = W_init_z; 
        if(GPU_mode)
            W_z_opt = gather(W_z_opt);
        end
    else
        W_z_opt = minFunc(@ZSL_ObjFunc_Wz, reshape(W_init_z,[m*d_z, 1]), options, num_Parts, m, d_x, d_z, W_init_x,...
            X, Z, Y, ZZ_t, D_xzi_cell, lambda1, lambda2, GPU_mode);
       
        W_z_opt = reshape(W_z_opt, [m, d_z]); 
        W_x_opt = W_init_x;
        if(GPU_mode)
            W_x_opt = gather(W_x_opt);
        end
    end
    trainWx_FLAG = ~trainWx_FLAG; %  train W_z and W_x alternatively 
    
    %%%% calculate each loss
    parts_Regu =0;
    if(lambda2)
        W_x_t = W_x_opt'; 
        for i = 1:num_Parts
            W_xz = W_x_t((d_p*(i-1)+1) : d_p*(i),:) * W_z_opt; 
            parts_Regu = parts_Regu + sum(sqrt(sum(W_xz.^2, 1))); 
        end
    end
    
    Wxt_Wz_Z =W_x_opt' * W_z_opt * Z;
    
    f0 = norm( (X'* Wxt_Wz_Z - Y) ,'fro')^2; 
    f1 = lambda1 * norm( Wxt_Wz_Z ,'fro')^2; 
    f2 = lambda2 * parts_Regu; 
    f =  f0 + f1 + f2; 
  
    fprintf('\nTime for loop: %f seconds.\n', etime(clock,t)); 
    fprintf('train_acc = %1.4f%%\n',  100 * (1-get_error(X, W_x_opt, W_z_opt , Z, y)));
    fprintf('Total Loss: f = %f,  Loss_0 = %f,  Loss_1 = %f,  Loss_2 = %f \n\n', f, f0, f1,f2); 
    
    fid = fopen([path.repath '/results.txt'], 'a+');
    fprintf(fid, 'ITER %d:     train_acc = %1.4f%%\n',  train_Itn, 100 * (1-get_error(X, W_x_opt, W_z_opt , Z, y)));
    fprintf(fid, 'Total Loss: f = %f,  Loss_0 = %f,  Loss_1 = %f,  Loss_2 = %f \n\n', f, f0, f1,f2); 
    fclose(fid);
    
    if(mod(train_Itn, 10) == 0)
        Weight_Name = sprintf([path.repath '/Weight_opt_%d'], train_Itn);  
        save(Weight_Name, 'W_x_opt', 'W_z_opt'); 
    end
    
    %%%% use the current weight as initialization. 
    W_init_z = W_z_opt; 
    W_init_x = W_x_opt; 
end

end

function err = get_error(X, W_x, W_z, Z, y)
    pred_score =X' * W_x' * W_z * Z;
    [~, maxIdx] = max(pred_score');
    pred_id = maxIdx';
    GT_id = y;
    err = sum(pred_id ~= GT_id) / length(y);
end








