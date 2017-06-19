clear all;
clc;

rng('default');
rng(14);

addpath('../data');
addpath('./helpers');
addpath('./MoCapToolbox_v1.4/mocaptoolbox');

%% Script config variables
plots = 0;
showVideo = 1;

%% Pattern constants
nP = 15;
pattDim = 61;

%% Appertures
alphas = 10 * ones(1,nP); % apertures
alphas(2) = 3;
alphas(13) = 10;

% Number of motion patterns
% 1 ExaggeratedStride 2 SlowWalk 3 Walk 4 RunJog 5 CartWheel  6 Waltz
% 7 Crawl  8 Standup  9 Getdown 10 Sitting  11 GetSeated  12 StandupFromStool
% 13 Box1  14 Box2 15 Box3

% Setting sequence order of patterns to be displayed
pattOrder = [10 12 2 1 4 1 6 3];
% Setting durations of each pattern episode (note: 120 frames = 1 sec)
pattDurations = [150 260 200 200 250 130 630 200];
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

% Set segment lenghts of stick figure to the ones found in the
% GetSeated data
segmentlengths = segLengthsGetSeated;

% set default height of centx   er of gravity to mean of some of the traces
% (needed to place visualized stick guy in a reasonably looking height
% above ground)
hmean = mean([hmeanExaStride, hmeanSlowWalk, ...
    hmeanWalk]);

% Pattern durations
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

% Back-distribute concatenated traces over the individual patterns
patts = cell(1,nP);
startInd = 1;
for i = 1:nP
    patts{i} = ...
        nnNormAllData(startInd:startInd + pattLengths(i)-1,:);
    startInd = startInd + pattLengths(i);
end;

%% Reservoir parameters
% Fixed testing and washout size
testSize = 1000;
washoutSize = 50;

% Reservoir size
resSize = 1000;

% Scaling parameters
wStarScale = 1;
wInScale = 1;
biasScale = .8;
wReg = 0.5;
wOutReg = 0.5;
a = 1;

%% Compute Reservoir
w0 = 2 * rand(resSize, resSize) - 1;
spectralRadW0 = max(abs(eigs(w0)));
wStar = w0 * ((1.00 / spectralRadW0) * wStarScale);

wIn = (2 * rand(resSize, pattDim) - 1) * wInScale;
bias = (2 * rand(resSize, 1) - 1) * biasScale;

%% Train the network
totalDataLength = sum(pattLengths);
totalLearnLength = totalDataLength - nP * washoutSize;

allTrainxArgs = zeros(resSize + 1, 0);
allTrainOldxArgs = zeros(resSize, 0);
allTrainWtargets = zeros(resSize, 0);
allTrainOuts = zeros(pattDim, 0);
Wtargets = zeros(resSize,0);
patternRs = cell(1,nP);
startXs = zeros(resSize, nP);
internalTraining = cell(1, nP);

for p = 1:nP
    % Get next pattern
    patt = patts{p};
    % Remove 5 and 17 channel for they are only noise
    patt(:, [5 17]) = zeros(pattLengths(p), 2);
    learnLength = pattLengths(p) - washoutSize;
    xCollector = zeros(resSize + 1, learnLength );
    xOldCollector = zeros(resSize, learnLength );
    WTargetCollector = zeros(resSize, learnLength);
    pCollector = zeros(pattDim, learnLength );
    x = zeros(resSize, 1);
    for n = 1:(washoutSize + learnLength)
        u = patt(n,:)'; 
        xOld = x;
        Wtarget = wStar * xOld + wIn * u;
        x = (1-a)*xOld + a * tanh(Wtarget + bias);
        if n == washoutSize
            startXs(:,p) = x;
        end
        if n > washoutSize
            xCollector(:, n - washoutSize ) = [x; 1];
            xOldCollector(:, n - washoutSize ) = xOld;
            WTargetCollector(:, n - washoutSize ) = Wtarget;
            pCollector(:, n - washoutSize) = u;
            
        end
        uOld = u;
    end
    patternRs{p} = xCollector(1:end-1,:) * xCollector(1:end-1,:)'...
        / learnLength;
    
    internalTraining{p} = xOldCollector(1:10, :);
    allTrainxArgs = [allTrainxArgs, xCollector];
    allTrainOldxArgs = [allTrainOldxArgs, xOldCollector];
    allTrainOuts = [allTrainOuts, pCollector];
    allTrainWtargets = [allTrainWtargets, WTargetCollector];
end

%% Compute pattern readout
wOut = (inv(allTrainxArgs * allTrainxArgs' + ...
    wOutReg * eye(resSize + 1)) ...
    * allTrainxArgs * allTrainOuts')';
% training error
outsRecovered = wOut*allTrainxArgs;
NRMSE_readout = nrmse(outsRecovered, allTrainOuts);
disp(sprintf('mean NRMSE readout: %g   mean abs Wout: %g',...
    mean(NRMSE_readout(not(isnan(NRMSE_readout)),1)),...
    mean(mean(abs(wOut)))));

outsTrain = cell(1,nP);
wo = washoutSize;
startInd = 1;
for i = 1:nP
    outsTrain{i} = outsRecovered(:,...
        startInd:startInd + pattLengths(i) - wo -1)';
    startInd = startInd + pattLengths(i) - wo;
end


%% Compute W
w = (inv(allTrainOldxArgs * allTrainOldxArgs' + ...
    wReg * eye(resSize)) * allTrainOldxArgs * allTrainWtargets')';
% training errors per neuron
NRMSE_W = nrmse(w*allTrainOldxArgs, allTrainWtargets);
disp(sprintf('mean NRMSE W: %g   mean abs W: %g ', ...
    mean(NRMSE_W), mean(mean(abs(w)))));


%% Compute conceptors
Cs = cell(4, nP);
for p = 1:nP
    [U S V] = svd(patternRs{p});
    Snew = (S * inv(S + alphas(p)^(-2) * eye(resSize)));
    C = U * Snew * U';
    Cs{1, p} = C;
    Cs{2, p} = U;
    Cs{3, p} = diag(Snew);
    Cs{4, p} = diag(S);
end

%% Generate test points for each pattern
internalTesting = zeros(testSize, 10, nP);
simpleTestData = zeros(pattDim, testSize, nP);

for p = 1:nP
    C = Cs{1, p};
    x = startXs(:, p);
    
    for n = 1:testSize
        xOld = x;
        x = (1-a)*xOld + a * tanh(w *  x + bias);
        
        internalTesting(n,:,p) = x(1:10,1)';
        x = C * x;
        simpleTestData(:,n,p) = wOut * [x; 1];
    end;
end;

%% Plots
% The plots of which pattern to show
patternNumber = 13;

internalLen = size(internalTraining{patternNumber}, 2);
thisPatt = patts{patternNumber}';
pattLen = size(thisPatt, 2);

% figure('units','normalized','position',[.1 .1 .8 .4]);
figure();
subplot(3,1,1);
plot(internalTraining{patternNumber}');
subplot(3,1,2);
plot(internalTesting(1:internalLen,:,patternNumber));
subplot(3,1,3);
plot(internalTraining{patternNumber}' - internalTesting(1:internalLen,:,patternNumber));

figure(2);
for i=1:pattDim
    subplot(8, 8, i);
    plot(thisPatt(i,:), 'b');
end;

figure(3);
for i=1:pattDim
    subplot(8, 8, i);
    plot(simpleTestData(i,:,patternNumber), 'b');
end;


%% Show stick figure video
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

p_CTestPLMorph = zeros(pattDim, L);
x = startXs(:,pattOrder(1));
for n = 1:L
    xOld = x;
    x = (1-a)*xOld + a * tanh(w *  x + bias);
    % find which mu indices are not 0
    thismu = mus(:,n);
    allInds = (1:nP)';
    muNot0Inds = allInds(thismu ~= 0);
    if length(muNot0Inds) == 1
        thisC = Cs{1,muNot0Inds};
    else
        thisC = thismu(muNot0Inds(1)) * Cs{1,muNot0Inds(1)} + ...
            thismu(muNot0Inds(2)) * Cs{1,muNot0Inds(2)};
    end
    x = thisC * x;
    p_CTestPLMorph(:,n) = wOut * [x; 1];
end

j2spar.type = 'j2spar';
j2spar.rootMarker = 1;
j2spar.frontalPlane = [6 10 2 ];
j2spar.parent = [0 1 2 3 4 1 6 7 8 1 10 11 11 13 14 15 11 17 18 19];
j2spar.segmentName = cell(1,19);

nnOutnorm = p_CTestPLMorph';

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
%japarVideo.scrsize = [400 300];
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

