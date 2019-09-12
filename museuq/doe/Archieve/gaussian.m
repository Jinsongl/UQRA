function [y] = gaussian(xx, u, a)

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% GAUSSIAN PEAK INTEGRAND FAMILY
%
% Authors: Sonja Surjanovic, Simon Fraser University
%          Derek Bingham, Simon Fraser University
% Questions/Comments: Please email Derek Bingham at dbingham@stat.sfu.ca.
%
% Copyright 2013. Derek Bingham, Simon Fraser University.
%
% THERE IS NO WARRANTY, EXPRESS OR IMPLIED. WE DO NOT ASSUME ANY LIABILITY
% FOR THE USE OF THIS SOFTWARE.  If software is modified to produce
% derivative works, such modified software should be clearly marked.
% Additionally, this program is free software; you can redistribute it 
% and/or modify it under the terms of the GNU General Public License as 
% published by the Free Software Foundation; version 2.0 of the License. 
% Accordingly, this program is distributed in the hope that it will be 
% useful, but WITHOUT ANY WARRANTY; without even the implied warranty 
% of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU 
% General Public License for more details.
%
% For function details and reference information, see:
% http://www.sfu.ca/~ssurjano/
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% INPUTS:
%
% xx = [x1, x2, ..., xd]
% u  = [u1, u2, ..., ud] (optional), with default value
%      [0.5, 0.5, ..., 0.5]
% a  = [a1, a2, ..., ad] (optional), with default value [1, 1, ..., 1]
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[m,d] = size(xx);

if (nargin == 1)
%     u = repmat(0.5, 1, d);
    u = [1,0.5];
    a = ones(1, d);
elseif (nargin == 2)
    a = ones(1, d);
end

sum = zeros(m,1);
for im = 1 : m
    for ii = 1:d
        xi = xx(im,ii);
        ai = a(ii);
        ui = u(ii);
        new = ai^2 * (xi-ui)^2;
        sum(im) = sum(im) + new;
    end
end

y = exp(-sum);

end
