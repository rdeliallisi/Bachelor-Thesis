% Clear previous workspace
clear all;
clc;

rng('default');
rng(14);

addpath('../data');
addpath('./helpers');
addpath('./MoCapToolbox_v1.4/mocaptoolbox');

% Script config variables
plots = 1;
% The plots of which pattern to show
showVideo = 0;

% Pattern constants
nP = 3;
pattDim = 61;

% Number of motion patterns
% 1 ExaggeratedStride 2 SlowWalk 3 Walk 4 RunJog 5 CartWheel  6 Waltz
% 7 Crawl  8 Standup  9 Getdown 10 Sitting  11 GetSeated  12 StandupFromStool
% 13 Box1  14 Box2 15 Box3

% Setting sequence order of patterns to be displayed
pattOrder = [10 12 2 1 4 1 6 3 9 7 8 3 13 15 14 2  5 3 2 11 ];
% Setting durations of each pattern episode (note: 120 frames = 1 sec)
pattDurations = [ 150 260 200 200 250 130 630 200 120 400 100 100  ...
    250 400 300 150 670 100 150 300 ];
% Setting durations for morphing transitions between two subsequent patterns
pattTransitions = 120 * ones(1, length(pattOrder)-1);

% Load pattern data
load nnRawExaStride; load nnRawSlowWalk;  load nnRawWalk;
load nnRawRunJog; load nnRawCartWheel; load nnRawWaltz;
load nnRawCrawl; load nnRawStandup; load nnRawGetdown;
load nnRawSitting; load nnRawGetSeated; load nnRawStandupFromStool;
load nnRawBox1; load nnRawBox2;  load nnRawBox3;

% Assign mocap data to p(pattern) variables
p1 = nnRawDataExaStride; p2 = nnRawDataSlowWalk;
p3 = nnRawDataWalk; p4 = nnRawDataRunJog;
p5 = nnRawDataCartWheel; p6 = nnRawDataWaltz;
p7 = nnRawDataCrawl; p8 = nnRawDataStandup;
p9 = nnRawDataGetdown; p10 = nnRawDataSitting;
p11 = nnRawDataGetSeated; p12 = nnRawDataStandupFromStool;
p13 = nnRawDataBox1;  p14 = nnRawDataBox2; p15 = nnRawDataBox3;

% set segment lenghts of stick figure to the ones found in the
% GetSeated data
segmentlengths = segLengthsGetSeated;

% set default height of centx   er of gravity to mean of some of the traces
% (needed to place visualized stick guy in a reasonably looking height
% above ground)
hmean = mean([hmeanExaStride, hmeanSlowWalk, ...
    hmeanWalk]);

% pattern durations
pl1 = size(p1,1); pl2 = size(p2,1); pl3 = size(p3,1);
pl4 = size(p4,1); pl5 = size(p5,1); pl6 = size(p6,1);
pl7 = size(p7,1); pl8 = size(p8,1); pl9 = size(p9,1);
pl10 = size(p10,1); pl11 = size(p11,1); pl12 = size(p12,1);
pl13 = size(p13,1); pl14 = size(p14,1); pl15 = size(p15,1);

pattLengths = [pl1 pl2 pl3 pl4 pl5 pl6  pl7  pl8 ...
    pl9 pl10 pl11 pl12 pl13 pl14 pl15];

% Put all the patterns in a single matrix
allData = [p1; p2; p3; p4; p5; p6; p7; p8; p9; p10; p11; p12; p13; p14; p15];

% Normalize all the data
[nnNormAllData, scalings, shifts] = normalizeDataMinus1Plus1(allData);

% back-distribute concatenated traces over the individual patterns
patts = cell(1,nP);
startInd = 1;
for i = 1:nP
    patts{i} = ...
        nnNormAllData(startInd:startInd + pattLengths(i)-1,:);
    startInd = startInd + pattLengths(i);
end;

% Apertures
alphas = 5 * ones(1,nP);

% Input, reservoir, output dimensions
res_size = 1000;
expand_size = 10000;
in_dim = pattDim;
out_dim = pattDim;

% Testing/washout size
test_size = max(pattLengths(1:nP));
washout_size = 50;

% Scaling parameters
gf_scale = 1.6;
w_in_scale = 1;
bias_scale = 0.3;

% Regularizer
reg = 1e-2;

% Leaking rate
a = 0.6;

% Randomly initialize g and g from a uniform distribution(-1,1)
g = 2 * rand(res_size, expand_size) - 1;
f = 2 * rand(expand_size, res_size) - 1;

% Compute scaling parameters for g and f
prod = g * f;
spec_rad_prod = max(abs(eigs(prod)));
real_scale = sqrt(gf_scale / spec_rad_prod);

% Scale G and F
g = real_scale .* g;
f = real_scale .* f;

% Compute and scale w_in
w_in = 2 * rand(res_size, in_dim) - 1;
w_in = w_in_scale .* w_in;

% Compute and scale the bias
bias = 2 * rand(res_size, 1) - 1;
bias = bias_scale .* bias;

%% Containers for training
d_z = zeros(expand_size, sum(pattLengths) - nP * washout_size);
d_p = zeros(res_size, sum(pattLengths) - nP * washout_size);
w_r = zeros(res_size, sum(pattLengths) - nP * washout_size);
w_p = zeros(in_dim, sum(pattLengths) - nP * washout_size);
counter = 1;
start_r = zeros(res_size, nP);
start_z = zeros(expand_size, nP);
internalTraining = cell(1, nP);
internalTrainingExt = cell(1, nP);
c = cell(1, nP);
% Initialize reservoir state and expanded state
r = zeros(res_size, 1);
z = zeros(expand_size, 1);

for iteration = 1:nP
    fprintf('Iteration Train: %d\n', iteration);
    % Get the next pattern
    p = patts{iteration}';
    % Remove 5 and 17 channel for they are only noise
    p([5 17], :) = zeros(2,pattLengths(iteration));
    
    % Training data size
    train_size = pattLengths(iteration);
    
    % Conceptor data for single pattenr
    c_z = zeros(expand_size, train_size);
    patternTraining = zeros(train_size - washout_size,10);
    patternTrainingExt = zeros(train_size - washout_size, 10);
    for i = 1:train_size
        r_old = r;
        u = p(:, i);
        r = (1-a) * r + a * tanh(g * z + w_in * u + bias);
        z = f * r;
        if i == washout_size
            start_r(:, iteration) = r;
            start_z(:, iteration) = z;
        end;
        if i > washout_size
            d_z(:, counter)  = z_old;
            d_p(:, counter) = (w_in * u);
            w_r(:, counter) = r_old;
            w_p(:, counter) = u;
            c_z(:, i - washout_size) = z;
            patternTraining(i - washout_size, :) = r(1:10)';
            patternTrainingExt(i - washout_size, :) = z(1:10)';
            counter = counter + 1;
        end;
        
        z_old = z;
    end;
    
    internalTraining{iteration} = patternTraining;
    internalTrainingExt{iteration} = patternTrainingExt;
    % Compute random feature conceptor of pattern
    z_square = c_z.^2;
    square_mean = mean(z_square, 2);
    c{iteration} = square_mean .* (square_mean + alphas(iteration)^-2).^-1;
end;

%% Compute d using ridge regression
d = (d_z * d_z' + reg * eye(expand_size)) \ d_z * d_p';
d = d';

mean_abs_d = mean(mean(abs(d)));
vec_nrmse_d = nrmse(d * d_z, d_p);
nrmse_d = mean(vec_nrmse_d(not(isnan(vec_nrmse_d)),1));
fprintf('mean NRMSE D: %g   mean abs D: %g\n', ...
    nrmse_d, mean_abs_d);

%% Compute w_out using ridge regression
w_out = (w_r * w_r' + reg * eye(res_size)) \ w_r * w_p';
w_out = w_out';

mean_abs_w_out = mean(mean(abs(w_out)));
vec_nrmse_w_out = nrmse(w_out * w_r, w_p);
nrmse_w_out = mean(vec_nrmse_w_out(not(isnan(vec_nrmse_w_out)),1));

fprintf('mean NRMSE Wout: %g   mean abs Wout: %g\n', ...
    nrmse_w_out, mean_abs_w_out);

%% Generate test_size test points for each pattern
simpleTestData = zeros(pattDim, test_size, nP);
internalTesting = zeros(test_size, 10, nP);
internalTestingExt = zeros(test_size, 10, nP);

for iteration = 1:nP
    fprintf('Iteration Test: %d\n', iteration);
    r = start_r(:, iteration);
    z = start_z(:, iteration);
    
    for i = 1:test_size
        r = (1-a) * r + a * tanh(g * z + d * z + bias);
        o = w_out * r;
        z = c{iteration} .* (f * r);
        
        internalTesting(i,:,iteration) = r(1:10)';
        internalTestingExt(i,:,iteration) = z(1:10)';
        simpleTestData(:,i,iteration) = o;
    end;
end;

%% Plots
if plots == 1
    %% Pattern to be plotted
    patternNumber = 3;
    %% Set ploting length by looking at training data available
    internalLen = size(internalTraining{patternNumber}, 1);
    thisPatt = patts{patternNumber}';
    pattLen = size(thisPatt, 2);
    
    %% Internal States
    figure();
    subplot(3,1,1);
    plot(internalTraining{patternNumber});
    subplot(3,1,2);
    plot(internalTesting(1:internalLen,:,patternNumber));
    subplot(3,1,3);
    plot(log(abs(internalTraining{patternNumber} - internalTesting(1:internalLen,:,patternNumber))));
    
    %% Expansion State
    figure();
    subplot(3,1,1);
    plot(internalTrainingExt{patternNumber});
    subplot(3,1,2);
    plot(internalTestingExt(1:internalLen,:,patternNumber));
    subplot(3,1,3);
    plot(log(abs(internalTrainingExt{patternNumber} - internalTestingExt(1:internalLen,:,patternNumber))));
    
    %% Original Pattern
    figure();
    for i=1:pattDim
        subplot(8, 8, i);
        plot(thisPatt(i,:), 'r');
    end;
    
    %% Generated Pattern
    figure();
    for i=1:pattDim
        subplot(8, 8, i);
        plot(simpleTestData(i,1:pattLen,patternNumber), 'b');
    end;
    
    %% RFCs
    figure();
    plot(1:expand_size, sort(c{patternNumber}), 'o');
end;

%% Show stick figure video
if showVideo == 1
    % Create morph sequence data
    L = sum(pattDurations) + sum(pattTransitions);
    mus = zeros(nP,L);
    
    for window = 1:length(pattOrder)
        if window == 1 % no transition
            mus(pattOrder(1),1:pattDurations(1)) = ...
                ones(1,pattDurations(1));
            startT = pattDurations(1) + 1;
        else
            % start with transition
            mus(pattOrder(window-1),...
                startT:startT+pattTransitions(window-1)-1) = ...
                (pattTransitions(window-1):-1:1) / pattTransitions(window-1);
            mus(pattOrder(window),...
                startT:startT+pattTransitions(window-1)-1) = ...
                (1:pattTransitions(window-1)) / pattTransitions(window-1);
            startT = startT + pattTransitions(window-1);
            mus(pattOrder(window),...
                startT:startT+pattDurations(window)-1) = ...
                ones(1,pattDurations(window));
            startT = startT + pattDurations(window);
        end
    end
    mus  = smoothmus( mus );
    
    morphData = zeros(pattDim, L);
    
    r = start_r(:, pattOrder(1));
    z = start_z(:, pattOrder(1));
    for n = 1:L
        r = (1-a) * r + a * tanh(g * z + d * z + bias);
        
        % find which mu indices are not 0
        thismu = mus(:,n);
        allInds = (1:nP)';
        muNot0Inds = allInds(thismu ~= 0);
        
        if length(muNot0Inds) == 1
            thisC = c{muNot0Inds};
        else
            thisC = thismu(muNot0Inds(1)) * c{muNot0Inds(1)} + ...
                thismu(muNot0Inds(2)) * c{muNot0Inds(2)};
        end;
        
        o = w_out * r;
        morphData(:,n) = o;
        z = thisC .* (f * r);
    end;
    
    j2spar.type = 'j2spar';
    j2spar.rootMarker = 1;
    j2spar.frontalPlane = [6 10 2 ];  % be careful about order
    j2spar.parent = [0 1 2 3 4 1 6 7 8 1 10 11 11 13 14 15 11 17 18 19];
    j2spar.segmentName = cell(1,19);
    
    nnOutnorm = morphData';
    
    nnRawDataRecovered = ...
        nnOutnorm * diag(1 ./ scalings) - ...
        repmat(shifts, size(nnOutnorm,1), 1);
    
    freq = 120;
    
    djrecovered = ...
        nnRaw2jHJ_absH(nnRawDataRecovered, hmean, segmentlengths, ...
        j2spar, freq);
    
    djrecovered.mus = mus;
    
    japarVideo.showmnum = 0;
    japarVideo.animate = 1;
    japarVideo.colors = 'wkkkk';
    japarVideo.trl = 0;
    japarVideo.numbers = [];
    japarVideo.cwidth = 5;
    japarVideo.twidth = 1;
    japarVideo.az = 20;
    japarVideo.el = 20;
    japarVideo.fps = 30;
    japarVideo.limits = [];
    japarVideo.scrsize = [400 350];
    japarVideo.showfnum = 0;
    japarVideo.conn2 = [];
    japarVideo.conn = [11 12; 13 11; 11 17; 13 14; 14 15; 15 16;...
        17 18; 18 19; 19 20; 10 11; 1 10; 1 2; 1 6; 2 3; 3 4; 4 5;...
        6 7; 7 8; 8 9];
    japarVideo.msize = 12;
    japarVideo.fontsize = 12;
    japarVideo.folder = '0'; % set to string '0' if no save is wanted, else
    % give a name for a folder where frame png
    % files are to be stored
    japarVideo.center = 1;
    japarVideo.pers.c = [0 -10000 0];
    japarVideo.pers.th = [0 0 0];
    japarVideo.pers.e = [0 -6000 0];
    japarVideo.nGridlines = 5;
    japarVideo.botWidth = 5000;
    
    japarVideo.fignr = 11;
    djrecoveredResampled = mcresampleHJ(djrecovered, japarVideo.fps);
    xRootShifts = djrecoveredResampled.data(2:end,1) - ...
        djrecoveredResampled.data(1:end-1,1);
    yRootShifts = djrecoveredResampled.data(2:end,2) - ...
        djrecoveredResampled.data(1:end-1,2);
    
    djrecoveredCentered = mccenterxyFrames(djrecovered);
    djrecoveredCenteredResampled = ...
        mcresampleHJ(djrecoveredCentered, japarVideo.fps);
    
    mcanimateHJ(djrecoveredCenteredResampled, japarVideo, 1,...
        xRootShifts, yRootShifts);
end;


