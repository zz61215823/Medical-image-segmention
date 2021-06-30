function rho = create_rho(im,io,ml,rf)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%  This function is a part of the implementation of the following paper:
%
% A. Pratondo, C.-K. Chui, and S.-H. Ong, 
% 揜obust Edge-Stop Functions for Edge-Based Active Contour Models in Medical Image Segmentation,? 
% IEEE Signal Processing Letters, vol. 23, no. 2, pp. 222 - 226, 2016 
%
% -----------------------------------------------------------------------------------------------------------
% INPUT 
% 		im : marked image (blue and red , indicating background and foreground)
%            Please note that blue marks should be 0000FF and red mark FF0000
% 		io : image wtihout marking
% 		ml : machine learning algorithm, 1 for k_NN, 2 for SVM
%       rf : rho function type, 1 for quadratic function , 2 for cosine function 
%  OUTPUT
%		rho : rho map. Later, it will be used to regularize g, g_new = g.* rho
%
% Author: Agus Pratondo
% E-mail: pratondo@gmail.com   
%         agus.praotndo.id@ieee.org  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
f = find(im(:,:,1) == 255);        	% foreground, it is a channel with red  value = 255
b = find(im(:,:,3) == 255) ;       	 	% background, it is a channel with blue value = 255


%% CREATE TRAINING DATA. A feature vector, generated from a 3x3 image patch, is used for all the experiments.
% 创建培训数据。 从3x3图像补丁生成的特征向量用于所有实验。
% The feature vector is stored in trX and its class in trY.
% 特征向量存储在trX中，其类存储在trY中

trX = zeros(length(b)+length(f),9); 	% training data (features)
trY = zeros(length(b)+length(f),1);		% training data (class / label)

% refine training data for background tiqu背景训练数据
for i = 1: length(b)
    [h,w] = ind2sub(size(io),b(i));
    trX(i,:) = [io(h-1,w-1) io(h-1,w) io(h-1,w+1) ...
	            io(h,w-1)   io(h,w)   io(h,w+1) ...
				io(h+1,w-1) io(h+1,w) io(h+1,w+1)];
    trY(i) = 0;							% class = 0 for the background
end

% refine training data for foreground 优化前景的训练数据
for j = 1: length(f)
    [h,w] = ind2sub(size(io),f(j));
    trX(i+j,:) = [io(h-1,w-1) io(h-1,w) io(h-1,w+1) ...
	              io(h,w-1)   io(h,w)   io(h,w+1) ...
				  io(h+1,w-1) io(h+1,w) io(h+1,w+1)];
    trY(i+j,:) = 1;						% class = 1 for the foreground
end

%% CREATE TESTING DATA.  It computes overall pixels (may duplicate w/ training data, but it's OK)
%  创建测试数据。 它可以计算总像素（可以与训练数据重复，但可以）
[h,w] =size(io);
M = io;
c1 = M(1:h-2   , 1:w-2  );
c2 = M(1:h-2   , 2:w-1  );
c3 = M(1:h-2   , 3:w    );
c4 = M(2:h-1   , 1:w-2  );
c5 = M(2:h-1   , 2:w-1  );
c6 = M(2:h-1   , 3:w    );
c7 = M(3:h     , 1:w-2  );
c8 = M(3:h     , 2:w-1  );
c9 = M(3:h     , 3:w    );
tsX = [c1(:) c2(:) c3(:) c4(:) c5(:) c6(:) c7(:) c8(:) c9(:)];

%% FIND SCORES
score =  ones(h,w);		
switch ml
	case 1 %ml is knn
		model = ClassificationKNN.fit(trX,trY,'NumNeighbors',24);
		[~,score] = predict(model,tsX);
        score = score(:,2);
	case 2 % ml is svm
        SVMModel = fitcsvm(trX,trY);
        CompactSVMModel = compact(SVMModel);
        CompactSVMModel = fitPosterior(CompactSVMModel,trX,trY);
        [~,PostProbs] = predict(CompactSVMModel,tsX);
        score = PostProbs(:,2);
end

s =  ones(h,w);						% initialization
s(2:h-1,2:w-1) = reshape(score,[h-2,w-2]);
if rf==1
% rho = 4*(s-0.5).^2;  % based on Eq. 9
rho=log(2*exp(4*(s-0.5).^2)-1);
% rho = s;
else
    rho = (cos(pi*s)).^2;                % based on Eq. 10 for p = 2 
end
end


