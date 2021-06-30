clear all;
close all;
tic;
global recim1     % record all segemented images, used for stopping condition
global recim2     
%% ����ͼ��
%ԭͼ��
Img1= imread('Drishti_GS\Images\drishtiGS_086.png');% original image
Img1=imresize(Img1,.25);
figure;imshow(Img1);
Img=double(Img1);
% Img=Img(:,:,1);
% Mark ͼ��  images with marks (blue : background, red : foreground)
Img_rb1 = imread('Drishti_GS\Images\drishtiGS_086md.png');   %1�����̣�2���ӱ�
Img_rb2 = imread('Drishti_GS\Images\drishtiGS_086mc.png');
Img_rb1=imresize(Img_rb1,.25);
Img_rb2=imresize(Img_rb2,.25);

% GT ͼ�� ground truth image
Img_gt1 = imread('Drishti_GS\Images\drishtiGS_086_ODsegSoftmap.png');
Img_gt2 = imread('Drishti_GS\Images\drishtiGS_086_cupsegSoftmap.png');
Img_gt1=imresize(Img_gt1,.25);
Img_gt2=imresize(Img_gt2,.25);
Img_gt1=im2bw(Img_gt1);
Img_gt2=im2bw(Img_gt2);

%% Ԥ���� 
%RGBתHSL
[h,s,L]=rgb2hsl(Img);

%��̬��� morphological closing 
se = strel('disk',8);
gambarclose = imclose(L,se);
% figure,imshow(gambarclose),title('morphological closing ');
%��ֵ���� median filter 
halus=medfilt2(gambarclose,[20 20]);
Img=halus;
% Img=gambarclose;
% figure,imshow(Img),title('gambarclose filter ');



%% ����ѧϰ��KNN���� Machine learning method
ml = 1;                                         % Machine learning type:  0 : no machine learning applied (traditional ESF) 
                                                %                         1 : knn; 
                                                %                         2 : svm                                               
switch ml                                       % define 'capture and header image
    case 0  
        capture = 'liver_tra';
        header  = 'Traditional ESF'; 
    case 1
        capture = 'liver_knn' ;
        header1  = 'Robust ESF using k-NN';
        header2  = 'Robust ESF using k-NN';
    case 2
        capture = 'liver_svm' ;
        header  = 'Robust ESF using SVM';
end



% ��������parameter setting
iter_outer1        = 3;
iter_outer2        = 3;
sigma             = 8;
timestep          = 2;                          % time step
mu                = 0.2/timestep;               % coefficient of the distance regularization term R(phi)
iter_inner1        = 10; 
iter_inner2        = 16; 
lambda            = 6;                          % coefficient of the weighted length term L(phi)
alfa              = 4;                        % coefficient of the weighted area term A(phi)
epsilon           = 1.5;                        % papramater that specifies the width of the DiracDelta function
potentialFunction = 'double-well';              % default choice of potential function

recim1             = logical(zeros([size(Img) iter_outer1]));
recim2             = logical(zeros([size(Img) iter_outer2]));
%��˹�˾��
% G=fspecial('gaussian',15,sigma);
% Img_smooth=conv2(Img,G,'same'); %������Aͬ����С�ľ�����Ĳ���     % smooth image by Gaussiin convolution
% % figure,imshow(Img_smooth),title('gaussian ');
% % [Ix,Iy]=gradient(Img_smooth);
% Img_smooth=Img;
% [Ix,Iy]=gradient(Img_smooth);
% f=Ix.^2+Iy.^2;
G=fspecial('gaussian',15,sigma);
Img_smooth=conv2(Img,G,'same'); %������Aͬ����С�ľ�����Ĳ���     % smooth image by Gaussiin convolution
[Ix,Iy]=gradient(Img_smooth);
f=Ix.^2+Iy.^2;
g = 1./(1+f);                    %��ͳ�ı�Եֹͣ����            % the traditional edge stop function
% figure('Name',' Traditional map of g'),imshow(g,[]);                                 
imwrite(g,['g/' capture '_gold.bmp'],'bmp');    % save image, it is a map of g

% Knn�㷨�õ�������Ե��OD��
if ml ~= 0                                      % ml == 0 means the traditional edge stop function is used 
   rhotype = 1;                                 % rho function type, 1 for quadratic function���κ��� , 2 for cosine function���Һ���
   rho1 = create_rho(Img_rb1,Img_smooth,ml,rhotype) ;
   figure('Name','KNN Map of rho'),imshow(rho1,[]);
   imwrite(rho1,['g/' capture '_rho1.bmp'],'bmp');% save image of rho  
   gnew1 = g.*rho1;    % see Eq. 11 in the paper
   figure('Name','KNN Map of g_new'),imshow(gnew1,[]);
   imwrite(gnew1,['g/' capture '_gnew1.bmp'],'bmp');% save image of gnew   
   g = gnew1;
end
% Knn�㷨�õ�������Ե��OC��
if ml ~= 0                                      % ml == 0 means the traditional edge stop function is used 
   rhotype = 1;                                 % rho function type, 1 for quadratic function���κ��� , 2 for cosine function���Һ���
   rho2 = create_rho(Img_rb2,Img_smooth,ml,rhotype) ;
   figure('Name','KNN Map of rho'),imshow(rho2,[]);
   imwrite(rho2,['g/' capture '_rho2.bmp'],'bmp');% save image of rho  
   gnew2 = g.*rho2;    % see Eq. 11 in the paper
   figure('Name','KNN Map of g_new'),imshow(gnew2,[]);
   imwrite(gnew2,['g/' capture '_gnew2.bmp'],'bmp');% save image of gnew   
   g = gnew2;
end
% [x,y]=size(rho1);
% f1=zeros(x,y);
% for i=1:x
%     for j=1:y
%         if(rho1(i,j)~=1)
%             f1(i,j)=1;
%         end
%         if(rho2(i,j)~=1)
%             f1(i,j)=1;
%         end
%     end
% end
% f1=~f1;
%    figure(10);
%    imshow(f1);
%% ��ʼ����
%OD��ʼ����
c          = -2;
initphi1   = c*ones(size(Img_rb1(:,:,1)));
f          = find(Img_rb1(:,:,1) == 255);         % red  channel with value 255 for an initial contour ��ʼ����ֵΪ255�ĺ�ɫͨ��
initphi1(f)= -c;  
phi1       = initphi1;

%OC��ʼ����
c          = -2;
initphi2   = c*ones(size(Img_rb2(:,:,1)));
f          = find(Img_rb2(:,:,1) == 255);         % red  channel with value 255 for an initial contour ��ʼ����ֵΪ255�ĺ�ɫͨ��
initphi2(f)= -c;  
phi2       = initphi2;

h0=figure('Name','Inital contour');             % display the initial contour and save it
imagesc(Img,[0, 255]); axis off; axis equal; colormap(gray); 
hold on;  contour(phi1, [0,0], 'b','LineWidth',1); 
contour(phi2, [0,0], 'g','LineWidth',1);

%% start level set evolution ��ʼˮƽ���ݱ�
for n1=1:iter_outer1  
%     for n2=1:iter_outer2  
    t1=1;
    phi1 = drlse_edge(phi1, g, lambda, mu, alfa, epsilon, timestep, iter_inner1, potentialFunction,t1);
    t2=2;
    phi2 = drlse_edge(phi2, g, lambda, mu, alfa, epsilon, timestep, iter_inner2, potentialFunction,t2);

    if mod(n1,2)==0
        pause(0.1);
        h=figure(7);
        imagesc(Img,[0, 255]); 
        colormap(gray);
        hold on; 
        axis off; axis equal; 
        contour(phi1, [0,0], 'b','LineWidth',1);  
        contour(phi2, [0,0], 'g','LineWidth',1);
        iter_outer1= [ 'Outer iteration: ',num2str(n1) ];
        title(iter_outer1);
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        fname = ['capture/Result/086od_oc/' capture 'iter_' sprintf('%04d',n1) '.png'];
        saveas(h,fname);
%         hold off;
    end
    if isstop(n1,phi1)                            %break the evolution if the current result is identical with a result from 5 or more steps before
        break
    end      
end

imagesc(Img, [0, 255]);
colormap(gray);
hold on;
axis off,axis equal
contour(phi1, [0,0], 'b','LineWidth',1);  
contour(phi2, [0,0], 'g','LineWidth',1);
title(['evolution result:Outer iteration: ' num2str(n1) ]);


%% refine the zero level contour, see Li et al. [22] implementation.   
alfa=0;
iter_refine = 10;
phi1 = drlse_edge(phi1, g, lambda, mu, alfa, epsilon, timestep, iter_inner1, potentialFunction,t1);
phi2 = drlse_edge(phi2, g, lambda, mu, alfa, epsilon, timestep, iter_inner2, potentialFunction,t2);

finalLSF1=phi1;
finalLSF2=phi2;

phigt1 = double(Img_gt1)*2;phigt1(phigt1==0)=-2;%%%%%
phigt2 = double(Img_gt2)*2;phigt2(phigt2==0)=-2;%%%%%




% hold on;  contour(initphi1, [0,0], 'w','LineStyle',':','LineWidth',1.5);%�ף����̳�ʼ����
% hold on;  contour(initphi2, [0,0], 'k','LineStyle',':','LineWidth',1.5);%�ڣ��ӱ���ʼ����
hf = figure('Name','The segmentation results, ground truth, and initialization');
imagesc(Img1,[0, 255]); axis off; axis equal; colormap(gray); 
hold on;  contour(phigt1,   [0,0], 'w','LineStyle','-','LineWidth',1.5);%�ף�����GT����
hold on;  contour(phigt2,   [0,0], 'k','LineStyle','-','LineWidth',1.5);%�ڣ��ӱ�GT����
hf = figure('Name','The segmentation results, ground truth, and initialization');
imagesc(Img1,[0, 255]); axis off; axis equal; colormap(gray); 
% hold on;  contour(phi1,     [0,0], 'b','LineWidth',1.5);%�죺���̷ָ���
hold on;  contour(phi2,     [0,0], 'g','LineWidth',1.5);%�����ӱ��ָ���
% str1=['Final zero level contour, ', ' OD iterations:',num2str(iter_outer1*iter_inner1+iter_refine),' OC iterations:',num2str(iter_outer2*iter_inner2+iter_refine)];
% title(str1);
title('Final zero level contour');
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% fname = ['capture/Result/086od_oc/'  '086r' '.png'];
% saveas(hf,fname);

phibw1 = im2bw(phi1);
% hfd=figure,imshow(phibw1),title('OD segmentation result binary image');
% % fname = ['capture/Result/086od_oc/'  '086od' '.png'];
% saveas(hfd,fname);

% figure(9);
% meshc(phi1); % for a better view, the LSF is displayed upside down
% hold on;  contour(phi1, [0,0], 'b','LineWidth',1.5);
% str=['Final level set function, ', num2str(iter_outer1*iter_inner1+iter_refine), ' iterations'];
% title(str);
% axis on;

phibw2 = im2bw(phi2);
% hfc=figure,imshow(phibw2),title('OC segmentation result binary image');
% % figure('Name','Segmentation result (binary image) '), imshow(phibw);
% % fname = ['capture/Result/086od_oc/'  '086oc' '.png'];
% % saveas(hfc,fname); 
% 
% figure(10);
% meshc(phi2); % for a better view, the LSF is displayed upside down
% hold on;  contour(phi2, [0,0], 'g','LineWidth',1.5);
% str=['Final level set function, ', num2str(iter_outer2*iter_inner2+iter_refine), ' iterations'];
% title(str);
% axis on;

r1=eval_metrics(Img_gt1,phibw1);
r2=eval_metrics(Img_gt2,phibw2);

[Jaccard1,Dice1]=compute_index(Img_gt1,phibw1);%%%%
[Jaccard2,Dice2]=compute_index(Img_gt2,phibw2);%%%%

%% �����ֵͼ�����
c1=bwarea(phibw1);%����OD
c2=bwarea(phibw2);%����OC

% nn1=sprintf('The OD is  %2f  The OC is %2f',c1,c2);
% nn2=sprintf(' The OC is %2f',c2);
% msgbox(nn1)
% msgbox(nn2)
%%
Quantitative1=sprintf('The OD result:\n Sensitivity1=%g\n Specificity2=%g\n Accuracy1=%g\n Jaccard1=%g\n Dice1=%g\n Sod=%2f',...
    r1(1),r1(2),r1(3),Jaccard1,Dice1,c1)
Quantitative2=sprintf('The OC result:\n Sensitivity2=%g\n Specificity2=%g\n Accuracy2=%g\n Jaccard2=%g\n Dice2=%g\n Soc=%2f',...
    r2(1),r2(2),r2(3),Jaccard2,Dice2,c2)
% msgbox(Quantitative1);
% msgbox(Quantitative2);

%% ����CDR
% cdr=c2./(c1);
% if cdr<0.3||cdr>0.6
%     msgbox('Glaucoma Disease ');
% else 
%     msgbox('Healthy��'); 
% end
% nn=sprintf('The CDR is  %2f ',cdr);%����CDR
% msgbox(nn)

% rim=(1-di)-(1-dil);
% RDR=bwarea(rim)./(c2);
% % pause(2)
% nn1=sprintf('The RDR is  %2f ',RDR/2);%����RDR/2
% msgbox(nn1)

toc;