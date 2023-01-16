clear all;
close all;
clc;


load('Datasets/retinamnist/hashCodes/hashCodes_64.mat');
data = hashCodes_64;
N = length(data);

load('Datasets/retinamnist/hashCodes/features_64.mat');
features = features_64;
load('Datasets/retinamnist/retinamnist_labels/labels_train.mat');
labels = labels_train;


load('Datasets/retinamnist/hashCodes/hashCodes_test_64.mat');
data_test = hashCodes_test_64;
%data_test = data_test(best_q_idx,:);

load('Datasets/retinamnist/hashCodes/features_test_64.mat');
features_test = features_test_64;
%features_test = features_test(best_q_idx,:);

load('Datasets/retinamnist/retinamnist_labels/labels_test.mat');
labels_test = labels_test;
%labels_test = labels_test(best_q_idx,:);



n  = length(data_test);
R  = 10;              % Pick first R retrieved images

for l=1:n
    
    query_features  = features_test(l,:); 
    
    q_new = repmat(data_test(l,:),N,1);
    dist = xor(data, q_new);
    hamming_dist(l,:) = sum(dist,2);

    [~,Retrieved_Items_Index] = sort(hamming_dist(l,:),'ascend');
    Retrieved_Items_AT_R = Retrieved_Items_Index(1, 1:R);
    
    Retrieved_Items_AT_R_Features{l,:} = features(Retrieved_Items_AT_R, :);

    euclidian_dist = pdist2(query_features, Retrieved_Items_AT_R_Features{l,:});     
    decision_matrix = [Retrieved_Items_AT_R' euclidian_dist'];    
    Retrieved_Items_AT_R_Ranked{l,:} = sortrows( decision_matrix , 2 );     
    Retrieved_Items{l,:} = Retrieved_Items_AT_R_Ranked{l,:}(:,1);

    query_label  = labels_test(l,:); 
    Retrieved_Items_AT_R_Ranked_Labels{l,:} = labels(Retrieved_Items{l,:},:);
    
    diff{l,:} = ismember(Retrieved_Items_AT_R_Ranked_Labels{l,:}, query_label  , 'rows'); 
    if isempty( diff{l,:})
            diff{l,:} = 0;
    end
    num_nz(l,:) = nnz( diff{l,:}(:,1) );
    s{l,:} = size(diff{l,:}(:,1), 1);
    
    for j=1:s{l,:};
        
        % Cummulative sum of the true-positive elements
        CUMM{l,:} = cumsum(diff{l,:});          
        Precision_AT_K{l,:}(j,1) = ( CUMM{l,:}(j,1)  ) / j;              
        Recall_AT_K{l,:}{j,1} = ( CUMM{l,:}(j,1)  ) / (num_nz(l,:)); %?????????????                    
    end  
    
    acc(l,:) = num_nz(l,:) / s{l,:};   % accuracy of the best cluster 
    %avg_Precision(l,:) = sum(Precision_AT_K{l,:}(:,1) ) / s{l,:};
    avg_Precision(l,:) = sum(Precision_AT_K{l,:}(:,1)  .* diff{l,:}(:,1) ) / num_nz(l,:);
    avg_Precision(isnan(avg_Precision))=0;
    
 end
 
mAP = sum(avg_Precision(:,1)) / l;
avg_acc = mean(acc);

best_q_idx = find(avg_Precision > .6); 