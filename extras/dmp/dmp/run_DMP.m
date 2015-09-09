%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% DMP code based on "Learning and Generalization of Motor Skills by Learning from Demonstration"
% by Peter Pastor, Heiko Hoffmann, Tamim Asfour, and Stefan Schaal
% ICRA 2009
%
% Bruno Castro da Silva, bsilva@cs.umass.edu
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


function [finalX, finalT] = run_DMP( dmp, x0, goal, tau)
clf;
silent=true;

% Numerical integration parameters
nIter = 10;
subDt = dmp.dt / nIter; % numerical integration substep to integrate from one dt to the next

xVec     = [x0];   % computed vector of positions
xdotVec  = [0.0];  % computed vector of velocities
tVec     = [0.0];  % computed vector with times (based on dt)

x = xVec(1);
v = xdotVec(1);
t = 0; % current time in the trajectory (from 0 to tau)

% integrates up to tau (total movement duration)
while (tau - t > subDt)
    % numerical integration to find the position and velocity at the current time t
    for i=1:nIter
        s = dmp.calcPhase(dmp, t);
        
        % Eq 6 
        vDot = 1/dmp.tau* (dmp.K*(goal - x) - dmp.D*v - dmp.K*(goal - x0)*s + dmp.K*dmp.fEval(dmp, s));        
        % Eq 7
        xDot = v/tau;

        % integrates
        v = v + vDot*subDt;
        x = x + xDot*subDt;
        t = t + subDt;
    end
    
    xVec    = [xVec x];
    xdotVec = [xdotVec v/tau];
    tVec    = [tVec t];
end

if (~silent)
    fprintf('Generating DMP trajectory...\n\t-Original DMP was trained with a movement of %.2f seconds\n\t\tstarted at x=%.2f, ended at x=%.2f\n', dmp.tau, dmp.x0, dmp.goal);
    fprintf('\t-The new trajectory will last %.2fs\n\t\tstart at x=%.2f, end at x=%.2f\n\n', tau, x0, goal);
    fprintf('Showing original and predicted trajectories...\n\n');
end    

finalX = xVec(end);
finalT = tVec(end);

hold on;
givenTrajPlot     = plot(dmp.original_tDemo, dmp.original_xDemo, 'b-');
plot(dmp.original_tDemo(1), dmp.original_xDemo(1), 'ko', dmp.original_tDemo(end), dmp.original_xDemo(end), 'ko');

predictedTrajPlot = plot(tVec, xVec,  'r-');
plot(tVec(1), xVec(1), 'k*', tVec(end), xVec(end), 'k*');

legend([givenTrajPlot, predictedTrajPlot], 'Given Trajectory', 'DMP');
%axis equal;
xlabel('Time (s)');
ylabel('System position (e.g., angle of robot arm)');
hold off;

end

