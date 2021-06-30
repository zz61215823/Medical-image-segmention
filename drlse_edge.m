function phi = drlse_edge(phi_0, g, lambda,mu, alfa, epsilon, timestep, iter, potentialFunction,t)
%  This Matlab code implements an edge-based active contour model as an
%  application of the Distance Regularized Level Set Evolution (DRLSE) formulation in Li et al's paper:
%  Li等人的距离中的距离正则化水平集演化（DRLSE）公式的应用，此Matlab代码实现了基于边缘的活动轮廓模型：
%      C. Li, C. Xu, C. Gui, M. D. Fox, "Distance Regularized Level Set Evolution and Its Application to Image Segmentation", 
%        IEEE Trans. Image Processing, vol. 19 (12), pp.3243-3254, 2010.
%
%  Input:
%      phi_0: level set function to be updated by level set evolution
%             水平集功能将通过水平集演化来更新
%      g: edge indicator function 边缘指示器功能
%      mu: weight of distance regularization term 距离正则项权重
%      timestep: time step时间步
%      lambda: weight of the weighted length term 加权长度项的权重
%      alfa:   weight of the weighted area term 加权面积项的权重
%      epsilon: width of Dirac Delta function Dirac Delta函数的宽度
%      iter: number of iterations 迭代次数
%      potentialFunction: choice of potential function in distance regularization term. 
%              As mentioned in the above paper, two choices are provided: potentialFunction='single-well' or
%              potentialFunction='double-well', which correspond to the potential functions p1 (single-well) 
%              and p2 (double-well), respectively.%
%  Output:
%      phi: updated level set function after level set evolution
%      水平集演变后更新了水平集功能
% Author: Chunming Li, all rights reserved
% E-mail: lchunming@gmail.com   
%         li_chunming@hotmail.com 
% URL:  http://www.imagecomputing.org/~cmli/
ene=[];    %保存能量值
F=0;
phi=phi_0;
[vx, vy]=gradient(g);% [Fx,Fy]=gradient(F)，其中Fx为其水平方向上的梯度，Fy为其垂直方向上的梯度，
if t==1    
for k=1:iter
    phi=NeumannBoundCond(phi);%诺伊曼边界条件
    [phi_x,phi_y]=gradient(phi);
    s=sqrt(phi_x.^2 + phi_y.^2);
    smallNumber=1e-10;  
    Nx=phi_x./(s+smallNumber); % add a small positive number to avoid division by zero添加一个小的正数以避免被零除
    Ny=phi_y./(s+smallNumber);
    curvature=div(Nx,Ny);
    if strcmp(potentialFunction,'single-well')%势函数
        distRegTerm = 4*del2(phi)-curvature;  % compute distance regularization term in equation (13) with the single-well potential p1.
                                              %用单阱电势p1计算公式（13）中的距离正则项。
    elseif strcmp(potentialFunction,'double-well')
        distRegTerm=distReg_p2(phi);  % compute the distance regularization term in eqaution (13) with the double-well potential p2.
    else
        disp('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.');
    end        
    diracPhi=Dirac(phi,epsilon);
    areaTerm=diracPhi.*g; % balloon/pressure force
    edgeTerm=diracPhi.*(vx.*Nx+vy.*Ny) + diracPhi.*g.*curvature;
    phi=phi + timestep*(mu*distRegTerm + lambda*edgeTerm + alfa*areaTerm);

    F=mu*distRegTerm + lambda*edgeTerm + alfa*areaTerm;
% size(ene(k))
% size(sum(F))

    ene(k) = sum(sum(F));

save afile1.txt -ascii ene;
end
else 
    for k=1:iter
    phi=NeumannBoundCond(phi);
    [phi_x,phi_y]=gradient(phi);
    s=sqrt(phi_x.^2 + phi_y.^2);
    smallNumber=1e-10;  
    Nx=phi_x./(s+smallNumber); % add a small positive number to avoid division by zero添加一个小的正数以避免被零除
    Ny=phi_y./(s+smallNumber);
    curvature=div(Nx,Ny);
    if strcmp(potentialFunction,'single-well')
        distRegTerm = 4*del2(phi)-curvature;  % compute distance regularization term in equation (13) with the single-well potential p1.
    elseif strcmp(potentialFunction,'double-well')
        distRegTerm=distReg_p2(phi);  % compute the distance regularization term in eqaution (13) with the double-well potential p2.
    else
        disp('Error: Wrong choice of potential function. Please input the string "single-well" or "double-well" in the drlse_edge function.');
    end        
    diracPhi=Dirac(phi,epsilon);
    areaTerm=diracPhi.*g; % balloon/pressure force%区域特征
    edgeTerm=diracPhi.*(vx.*Nx+vy.*Ny) + diracPhi.*g.*curvature;%边缘特征
    phi=phi + timestep*(mu*distRegTerm + lambda*edgeTerm + alfa*areaTerm);

    F=mu*distRegTerm + lambda*edgeTerm + alfa*areaTerm;
%     size(ene(k))
% size(sum(F))

    ene(k) = sum(sum(F));

save afile2.txt -ascii ene;
end
end
 
function f = distReg_p2(phi)
% compute the distance regularization term with the double-well potential p2 in eqaution (16)
[phi_x,phi_y]=gradient(phi);
s=sqrt(phi_x.^2 + phi_y.^2);
a=(s>=0) & (s<=1);
b=(s>1);
ps=a.*sin(2*pi*s)/(2*pi)+b.*(s-1);  % compute first order derivative of the double-well potential p2 in eqaution (16)
dps=((ps~=0).*ps+(ps==0))./((s~=0).*s+(s==0));  % compute d_p(s)=p'(s)/s in equation (10). As s-->0, we have d_p(s)-->1 according to equation (18)
f = div(dps.*phi_x - phi_x, dps.*phi_y - phi_y) + 4*del2(phi);

function f = div(nx,ny)
[nxx,junk]=gradient(nx);  
[junk,nyy]=gradient(ny);
f=nxx+nyy;

function f = Dirac(x, sigma)
f=(1/2/sigma)*(1+cos(pi*x/sigma));
b = (x<=sigma) & (x>=-sigma);
f = f.*b;

function g = NeumannBoundCond(f)
% Make a function satisfy Neumann boundary condition
[nrow,ncol] = size(f);
g = f;
g([1 nrow],[1 ncol]) = g([3 nrow-2],[3 ncol-2]);  
g([1 nrow],2:end-1) = g([3 nrow-2],2:end-1);          
g(2:end-1,[1 ncol]) = g(2:end-1,[3 ncol-2]);  