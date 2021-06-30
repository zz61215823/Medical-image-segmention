function ret = isstop(n,bwphi)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This function is a part of the implementation of the following paper:
%
% A. Pratondo, C.-K. Chui, and S.-H. Ong, 
% “Robust Edge-Stop Functions for Edge-Based Active Contour Models in Medical Image Segmentation,” 
% IEEE Signal Processing Letters, vol. 23, no. 2, pp. 222 - 226, 2016.
%
% ----------------------------------------------------------------------------------------------------
% INPUT :
%      n     = n-th iteration
%      bwphi = binary image generated from n-th iteration.
% OUTPUT : 
%     ret, 0 : the current image is NOT identical with the image from 5 or more steps before.
%          1 : the current image IS     identical with the image from 5 or more steps before.
%              It converges and the iteration should be stopped.  
%
%
% Author: Agus Pratondo
% E-mail: pratondo@gmail.com   
%         agus.praotndo.id@ieee.org  
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
global recim								% record every image in each iteration

bwphi(bwphi>0)=1;							% convert the map of phi to binary image
bwphi(bwphi<0)=0;
bwphi = logical(bwphi);

ret = 0; 									% default, do not stop the iteration  
if n > 20 									% assumption, to skip checking at the begining 
        if (isequal(bwphi,recim(:,:,n-5))|| isequal(bwphi,recim(:,:,n-6))||isequal(bwphi,recim(:,:,n-7)))			% Check : Is current image == image form 5 steps or more before
            ret =1;
        end
end

recim(:,:,n)= bwphi;						% record the current binary image