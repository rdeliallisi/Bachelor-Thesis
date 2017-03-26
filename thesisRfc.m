% Clear previous workspace
clear all;
clc;

rng('default');
rng(14);

addpath('./data');
addpath('./helpers');
addpath('./MoCapToolbox_v1.4/mocaptoolbox');

% Script config variables
plots = 1;
% The plots of which pattern to show
patternNumber = 1;
showVideo = 0;

% Pattern constants
nP = 15;
pattDim = 61;

%%% Number of motion patterns
% 1 ExaggeratedStride 2 SlowWalk 3 Walk 4 RunJog 5 CartWheel  6 Waltz
% 7 Crawl  8 Standup  9 Getdown 10 Sitting  11 GetSeated  12 StandupFromStool
% 13 Box1  14 Box2 15 Box3

% Setting sequence order of patterns to be displayed
pattOrder = [10 12 2 1 4 1 6 3 9 7 8 3 13 15 14 2 5 3 2 11 ];
% Setting durations of each pattern episode (note: 120 frames = 1 sec)
pattDurations = [ 150 260 240 200 250 130 630 200 120 400 100 100 ...
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

% set default height of center of gravity to mean of some of the traces
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
alphas = 100 * ones(1,nP);
alphas(2) = 3;
alphas(13) = 20;

% Input, reservoir, output dimensions
res_size = 600;
expand_size = 3000;
in_dim = pattDim;
out_dim = pattDim;

% Training/testing/washout size
test_size = 700;

% Scaling parameters
gf_scale = 0.5;
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

% Containers for training
d_z = [];
d_p = [];
w_r = [];
w_p = [];
start_r = [];
start_z = [];
internalTraining = cell(1, nP);
c = cell(1, nP);
% Initialize reservoir state and expanded state
r = zeros(res_size, 1);
z = zeros(expand_size, 1);

for iteration = 1:nP
    % Get the next pattern
    p = patts{iteration}';
    % Remove 5 and 17 channel for they are only noise
    p([5 17], :) = zeros(2,pattLengths(iteration));
    
    % Training, testing and washout data size
    train_size = pattLengths(iteration);
    washout_size = 50;
    
    % Conceptor data for single pattenr
    c_z = [];
    for i = 1:train_size
        r_old = r;
        u = p(:, i);
        r = (1-a) * r + a * tanh(g * z + w_in * u + bias);
        z = f * r;
        if i == washout_size
            start_r = [start_r r];
            start_z = [start_z z];
        end;
        if i > washout_size
            d_z = [d_z z_old];
            d_p = [d_p (w_in * u)];
            w_r = [w_r r_old];
            w_p = [w_p u];
            c_z = [c_z z];
            internalTraining{iteration} = [internalTraining{iteration}; r(1:10,:)'];
        end;
        z_old = z;
    end;
    
    % Compute random feature conceptor of pattern
    z_square = c_z.^2;
    square_mean = mean(z_square, 2);
    c{iteration} = square_mean .* (square_mean + alphas(iteration)^-2).^-1;
end;

% Compute d using ridge regression
d = inv(d_z * d_z' + reg * eye(expand_size)) * d_z * d_p';
d = d';

mean_abs_d = mean(mean(abs(d)));
vec_nrmse_d = nrmse(d * d_z, d_p);
nrmse_d = mean(vec_nrmse_d(not(isnan(vec_nrmse_d)),1));

% Compute w_out using ridge regression
w_out = inv(w_r * w_r' + reg * eye(res_size)) * w_r * w_p';
w_out = w_out';

mean_abss_w_out = mean(mean(abs(w_out)));
vec_nrmse_w_out = nrmse(w_out * w_r, w_p);
nrmse_w_out = mean(vec_nrmse_w_out(not(isnan(vec_nrmse_w_out)),1));

% Generate test_size test points for each pattern
simpleTestData = zeros(pattDim, test_size, nP);
internalTesting = zeros(test_size, 10, nP);

for iteration = 1:nP
    r = start_r(:, iteration);
    z = start_z(:, iteration);
    
    for i = 1:test_size
        r = (1-a) * r + a * tanh(g * z + d * z + bias);
        o = w_out * r;
        z = c{iteration} .* (f * r);
        
        internalTesting(i,:,iteration) = r(1:10)';
        simpleTestData(:,i,iteration) = o;
    end;
end;

if plots == 1
    % Produce plots for the specified pattern
    internalLen = size(internalTraining{patternNumber}, 1);
    thisPatt = patts{patternNumber}';
    pattLen = size(thisPatt, 2);
    
    figure();
    subplot(3,1,1);
    plot(internalTraining{patternNumber});
    subplot(3,1,2);
    plot(internalTesting(1:internalLen,:,patternNumber));
    subplot(3,1,3);
    plot(internalTraining{patternNumber} - internalTesting(1:internalLen,:,patternNumber));
    
    figure();
    for i=1:pattDim
        subplot(8, 8, i);
        plot(thisPatt(i,:), 'r');
    end;
    
    figure();
    for i=1:pattDim
        subplot(8, 8, i);
        plot(simpleTestData(i,1:pattLen,patternNumber), 'b');
    end;
    
end;
