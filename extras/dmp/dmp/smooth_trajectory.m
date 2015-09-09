function [ret] = smooth_trajectory(traj)

desiredNumber_DemoPoints = 100;

% downsamples demo trajectory
xDemo = downsample(traj, floor(length(traj)/desiredNumber_DemoPoints));
tDemo = (1:length(xDemo))';

oriXDemo = xDemo*100;

xDemo = xDemo * 100; % coordinates are now in centimeters
xDemo = floor(xDemo); % ignores any change smaller than a centimeter to filter out sensor noise

% xDemo = xDemo * 1000; % coordinates are now in milimeters
% xDemo = floor(xDemo); % ignores any change smaller than a milimeters to filter out sensor noise
% xDemo = xDemo / 10; % coordinates are now in centimeters

sm = smooth(xDemo, 0.1, 'lowess');
plot(tDemo, oriXDemo, 'b-', tDemo, xDemo, 'g-', tDemo, sm, 'r-');

ret = xDemo;
pause;
end