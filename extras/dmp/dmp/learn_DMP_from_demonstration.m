%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DMP code based on "Learning and Generalization of Motor Skills by Learning from Demonstration"
% by Peter Pastor, Heiko Hoffmann, Tamim Asfour, and Stefan Schaal
% ICRA 2009
%
% Bruno Castro da Silva, bsilva@cs.umass.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%



function [dmp] = learn_DMP_from_demonstration( traj, K, D, nBases, widthBasis, typeOfBasis )
clf;

silent=true;

dmp = struct;
dmp.nBases = nBases;
dmp.K = K;
dmp.D = D;
dmp.basisType = typeOfBasis;

if (dmp.nBases < 1)
    error('Needs at least 1 basis function!');
end

% Sanitize input
if (~isvector(traj.xDemo)) || (~isvector(traj.xDemo))
    error('Input trajectory has to be a vector!');
end
if (~isrow(traj.xDemo))
    traj.xDemo = traj.xDemo';
end
if (~isrow(traj.tDemo))
    traj.tDemo = traj.tDemo';
end


% --------------- initializes basis functions ----------------------------------------------------
dmp.timeConst = -log(0.05);   % ensures 95% phase convergence at time t=tau (ie., at the end of movement)

if strcmpi(dmp.basisType, 'Gaussian')
    uniformSpread=true; % if false, distributes them exponentially instead
    
    if (dmp.nBases==1)
        centers = 0.5;
        widths  = 1/widthBasis;
    else        
        if (uniformSpread)
            % distributes Gaussian bases functions phi_i in phase space (from 1 down to 0) such that they are uniformly spaced
            % in time and with constant width
            spacing = 1 / (dmp.nBases-1);
            centers = 0:spacing:1;  % uniform spacing
            widths  = ones(1,dmp.nBases) * 1/widthBasis;  % constant basis width
        else
            % distributes Gaussian bases functions phi_i in phase space (from 1 down to 0) such that they are exponentially spaced
            % in time and decrease width appropriately
            spacing = 1 / (dmp.nBases-1);
            centers = exp( (-spacing * dmp.timeConst) * (dmp.nBases-1:-1:0) );
            widths  = 1 ./ (exp(-spacing * dmp.timeConst * (dmp.nBases-1:-1:0)) * widthBasis);
        end
    end
    
    dmp.bases = @(atX) exp(-widths .* (atX-centers).^2); % bases(x): given a point x, evaluates each basis function phi_i at that point
    dmp.normalize_features = true;
    
elseif strcmpi(dmp.basisType, 'Fourier')
    % Fourier basis: cos(0), cos(pi*x), cos(2*pi*x), cos(3*x*pi), ...
    dmp.nBases = dmp.nBases+1; % adds a constant basis to our series of univariate Fourier basis
    dmp.bases = @(atX) cos(pi * (0:dmp.nBases-1) * atX); % note how first feature will be constant: cos(pi*0*atX)=1
    dmp.normalize_features = false;
    
else
    error('Unknown basis function type ' + dmp.basisType);
end

%-------------------------------------------------------------------------------------------------


% -------- test code: plots basis functions ------------------------------------------------------
% points = 0:0.01:1;
% vals = evalPointsAtBasisFunctions(dmp, points);
% hold on
% for i=1:dmp.nBases
%     plot(points, vals(:,i), '-');
% end
% xlabel('Time (s)');
% ylabel('Basis function activation');
% hold off;
% fprintf('Showing basis functions; press anything to continue...\n\n');
% pause();
%-------------------------------------------------------------------------------------------------



% -------- Learn from demonstration --------------------------------------------------------------

dmp.original_tDemo = traj.tDemo;
dmp.original_xDemo = traj.xDemo;

nSamples = size(traj.xDemo,2);
dmp.x0   = traj.xDemo(1);
dmp.goal = traj.xDemo(end);
dmp.tau  = traj.tDemo(end);
dmp.dt   = traj.tDemo(2) - traj.tDemo(1);

if (~silent)
    fprintf('Learning from demonstration...\n\t%d points given, total movement time %.2fs;\n\tstarts at x=%.2f, ends at x=%.2f\n\n', nSamples, dmp.tau, dmp.x0, dmp.goal);
end  

vDemo    = [0.0];
vdotDemo = [0.0];

vDemo2    = [0.0];
vdotDemo2 = [0.0];

% Initializes vector with velocities and accelerations, given positions from the demonstration
for i=2:nSamples
    dx = traj.xDemo(i) - traj.xDemo(i-1);
    dt = traj.tDemo(i) - traj.tDemo(i-1);
    
    vDemo    = [vDemo    (2*dx)/(dmp.dt/dmp.tau) - vDemo(i-1) ];
    vdotDemo = [vdotDemo (vDemo(i)-vDemo(i-1))/dt ];
    
end

dmp.original_vDemo    = vDemo;
dmp.original_vdotDemo = vdotDemo;

% Raw input to regression; regressor will compute features over these inputs (see getFeatures)
phaseVec = calcPhase(dmp, traj.tDemo);

% Desired output to regression
fTargets = [];
for i=1:nSamples
    % Eq. 8
    tmp = ((dmp.tau*vdotDemo(i) + dmp.D*vDemo(i)) / dmp.K) - (dmp.goal-traj.xDemo(i)) + ((dmp.goal-dmp.x0)*phaseVec(i));
    fTargets = [fTargets tmp];
end

dmp.weights = leastSquareWeights(dmp, phaseVec, fTargets');  % leastSquares computes the feature value f_i(x) each phase value x in X
dmp.weights;

% initializes methods of the class
dmp.evalPointsAtBasisFunctions = @evalPointsAtBasisFunctions;
dmp.fEval = @fEval;
dmp.calcPhase = @calcPhase;
dmp.getFeatures = @getFeatures;

end






%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%   HELPER FUNCTIONS   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


% --------------------------------------------------
function res = evalPointsAtBasisFunctions(dmp, points)
% Evaluates each basis function at each of the given points=[p0 p1 ...]
% Specifically, returns a (PxN) matrix
% res = [phi_0(p0) phi_1(p0) ...;
%        phi_0(p1) phi_1(p1) ...
%       ]
%     =  [bases(p0);
%         bases(p1); ...
%        ]
% Note that bases(x) returns the value of each basis phi_i when evaluated at x
%       and phi_i(x) is the i-th basis function, evaluated at x

nPoints = size(points,2);
res=zeros(nPoints, dmp.nBases);
for i=1:nPoints
    res(i,:) = dmp.bases(points(i));
end
end

% --------------------------------------------------
function res = fEval(dmp, s)
% evaluates f(s), according to Eq. 3
features = getFeatures(dmp, s);
res = features * dmp.weights;
end


% --------------------------------------------------
function res = calcPhase(dmp, curTime)
% solution to Eq. 4 (ie, that equation evaluated at s=curTime)
res = exp(- (dmp.timeConst/dmp.tau) * curTime );
end

% --------------------------------------------------
function res = getFeatures(dmp, s)
% Remember that we want to minimize J = sum_s (f_target(s) - f(s))^2
% where s_t is the phase variable at the t-th timestep

% If we are using Gaussian basis functions phi_i,
%         f(s) = (sum_i w_i * phi_i(s) * s) / sum_i phi(s)
%              = sum_i (w_i * f_i(s))
% where f_i(s) = (phi_i(s) * s) / sum_j phi_j(s)
%
% This function returns f_i(s) for each basis i
% Specifically, for a given point s it returns
%   [(phi_0(s) * s) / (sum_j phi_j(s));   (phi_1(s) * s) / (sum_j phi_j(s);  ...]
% = [f_0(s); f_1(s); ...]

% If we are using Fourier basis functions phi_i, note that we don' normalize
%         f(s) = (sum_i w_i * phi_i(s) * s)
%              = sum_i (w_i * f_i(s))
% where f_i(s) = (phi_i(s) * s)
%
% This function returns f_i(s) for each basis i
% Specifically, for a given point s it returns
%   [(phi_0(s) * s);   (phi_1(s) * s);  ...]
% = [f_0(s); f_1(s); ...]

% Remember that bases(x) gives the value of each basis when evaluated at point x

if (dmp.normalize_features)
    normalization = sum(dmp.bases(s)); % ie, sum_j phi_j(s)
else
    normalization = 1.0;
end

% This basically returns each term in the summation of Eq. 3 but without multiplying by the weight w_i
res = (s .* dmp.bases(s)) ./ normalization;
%fprintf('get features for point %f = [ %s]\n', s, sprintf('%.4f  ', res));

end

% --------------------------------------------------
function w = leastSquareWeights(dmp, X, y)

% Input parameters:
% X = [s0 s1 s2, ...], where s_i corresponds to the value of the phase variable at the i-th timestep
% Y = [fTarget(s0); fTarget(s1); ...]

% This function minimizes J = sum_s (f_target(s) - f(s))^2
% where    f(s) = (sum_i w_i * phi_i(s) * s) / sum_i phi(s)
%               = sum_i (w_i * f_i(s))
% where  f_i(s) = (phi_i(s) * s) / sum_j phi_j(s)

% See last paragraph, 1st column, page 2

% The system of equations that solves this is:
%
% w0 * f_0(s0) + w1 * f_1(s0) + ... = fTarget(s0)
% w0 * f_0(s1) + w1 * f_1(s1) + ... = fTarget(s1)
% ...
%
% which is equivalent to Aw=y
% where
%
% A = [f_0(s0) f_1(s0) ...;
%      f_0(s1) f_1(s1) ...
%     ]
%   = [getFeatures(s0);
%      getFeatures(s1);
%      ...
%      ]
% w = [w0; w1; ....]
% y = [fTarget(s0); fTarget(s1); ...]
%
% The solution is w = pinv(A' * A) * A' * y

nPoints = size(X, 2);

A = zeros(nPoints, dmp.nBases);

for i=1:nPoints
    A(i,:) = getFeatures(dmp, X(i));
end

w = pinv(A' * A) * A' * y;

end
