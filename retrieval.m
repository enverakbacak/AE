
clear all;
close all;
clc;


load('./Datasets/retinamnist/hashCodes/hashCodes_256.mat');
data = hashCodes_256;
N = length(data);

load('./Datasets/retinamnist/hashCodes/features_256.mat');
features = features_256;

load('./Datasets/retinamnist/labels/labels_train.mat');
labels = labels_train;


load('./Datasets/retinamnist/hashCodes/hashCodes_test_256.mat');
data_test = hashCodes_test_256;


load('./Datasets/retinamnist/hashCodes/features_test_256.mat');
features_test = features_test_256;


load('./Datasets/retinamnist/labels/labels_test.mat');
labels_test = labels_test;


hR = 20; % Hamming Radious;
i  = 1;
n  = length(data_test);
%k  = 1000;

for l = i:n
    

    query                       = repmat(data_test(l,:),N,1);
    dist                        = xor(data, query);
    hamming_dist                = sum(dist,2);

    hamming_dist_hR             = hamming_dist <= hR;    % Hamming Radious
    r_index                     = find(hamming_dist_hR); % Image Indexex satisfying hR
    
    r_features                  = features(r_index, :);    % Features 
    euclidian_dist              = pdist2(features_test,  r_features ); % Euclidean dists for reranking
    euclidian_dist              =  euclidian_dist';

    decision_matrix             = [r_index euclidian_dist];  
    decision_matrix_sorted      = sortrows(decision_matrix, 2); 
    Retrieved_Items             = decision_matrix_sorted(:, 1);
    
    query_label(l,:)            = labels_test(l ,:);

    Retrieved_Items_Labels      = labels(Retrieved_Items,:);
    
    diff     = ismember(Retrieved_Items_Labels, query_label(l,:)   , 'rows'); 
    if isempty( diff)
            %% 
            diff = 0;
    end

    num_nz(l,:) = nnz( diff(:,1) );
    s{l,:} = size(diff(:,1), 1);
    
    for j=1:s{l,:};
        
        CUMM{l,:} = cumsum(diff);          
        Precision_AT_K{l,:}(j,1) = ( CUMM{l,:}(j,1)  ) / j;              
        Recall_AT_K{l,:}{j,1} = ( CUMM{l,:}(j,1)  ) / (num_nz(l,:)); %                
    end  
    
    acc(l,:) = num_nz(l,:) / s{l,:};   
    avg_Precision(l,:) = sum(Precision_AT_K{l,:}(:,1)  .* diff(:,1) ) / num_nz(l,:);
    avg_Precision(isnan(avg_Precision))=0;
    
 end
 

mAP       = sum(avg_Precision(:,1)) /(n-i+1);
ACC       = sum(acc(:,1)) / (n-i+1);

best_q_idx_256 = find(avg_Precision > .6); 
