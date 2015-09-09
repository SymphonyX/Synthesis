function run(pK,pD)

% ------- loads an example trajectory
desiredTau = 5;    % total duration of synthetic training movement
desiredDT  = 0.02;

%[tDemo, xDemo] = fakeTraj(1.0, 80, desiredTau, desiredDT );
%[tDemo, xDemo] = high5_joint2( desiredDT );
[tDemo, xDemo] = high5_joint5( desiredDT );

% plot(tDemo, xDemo, 'k-');
% pause;
traj = struct;
traj.tDemo = tDemo;
traj.xDemo = xDemo;

% sm = smooth(traj.xDemo, 30);
% plot(traj.tDemo, traj.xDemo, 'b-', traj.tDemo, sm, 'g-');
% pause;


% ------ learns a DMP from demonstration of trajectory
K         = 200.0;  % larger, less delay in going up
D         = 50.0;  % smaller helps getting closer to the goal at the end
K=pK;
D=pD;
nBases    = 5;
baseWidth = 0.1;
basisType = 'Gaussian';
%basisType = 'Fourier';
dmp = learn_DMP_from_demonstration(traj, K, D, nBases, baseWidth, basisType);

% ----- re-executes the DMP with different meta-parameters
x0    = traj.xDemo(1);   % reparametrization of desired initial state
%x0 = 20;
%x0    = 40;
goal  = traj.xDemo(end); % reparametrization of desired goal state
%goal = 10;
%goal  = 0;
tau   = traj.tDemo(end); % reparametrization of desired movement duration
%tau = 1.0;
%tau   = 4;

% x0=   1.4503
% goal=  2.1793
% tau=  2.3437



[finalPos, finalT] = run_DMP( dmp, x0, goal, tau)
fprintf('done\n');


end

