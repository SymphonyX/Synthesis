function [ tVec, xVec ] = fakeTraj( amplitude, tChange, totalT, dt )

% K=200, D=50, Scott's differentiation
% K=5, D=1, Scott's differentiation
% K=125, D=4, new differentiation

%initial_x = rand(1)*50 - 25; % rand initial position in [-25,25]
initial_x = -10;

tVec = [0.0];
xVec = [initial_x];

curT = 0.0;
step = 0;

curT = curT + dt;
step = step + 1;

x = initial_x;
v = 0.0; % assume initial velocity is zero

while (curT <= totalT)
    
    if (step < tChange)
        curAmplitude = amplitude;
    else
        curAmplitude = -amplitude*2;
    end
    
    % numerically integrate this step
    nIter = 100;
    dtSlice = dt/nIter;
    for i=1:nIter
        x = x + v*dtSlice;
        v = v + curAmplitude*dtSlice;
    end
    
    xVec = [xVec x];
    tVec = [tVec curT];
    
    curT = curT + dt;
    step = step + 1;
end

%plot(tVec, xVec, '-');

end

