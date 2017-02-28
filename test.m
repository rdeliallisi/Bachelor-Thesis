clear all;
clc;

rand('seed', 14);

addpath('./data');

load nnRawWalk.mat;

d = nnRawDataWalk';

% Input, reservoir, output size
res_size = 600;
in_size = 0;
out_size = 61;

% Training data size and washout time
test_size = 320;
train_size = 320;
washout_size = 50;

% Scaling parameters
w_scale = 1.7;
w_back_scale = 0.05;
w_in_scale = 1;
bias_scale = 1.0;
reg = 1e-2;

% Leaking rate
a = 0.6;

% Scale internal weights w
w_0 = 2 * rand(res_size, res_size) - 1;
spectral_rad_w_0 = max(abs(eigs(w_0)));
w = w_0 * ((1.00 / spectral_rad_w_0) * w_scale);

% Compute w_back
w_back = (2 * rand(res_size, out_size) - 1) * w_back_scale;

% Compute bias
bias = (2 * rand(res_size, 1) - 1) * bias_scale;

% Internal state collector m and teacher collector t
m = zeros(train_size - washout_size, res_size);
t_T = d';
t = t_T(washout_size + 1:train_size,:);

% x column vector
x = zeros(res_size, 1);
o = zeros(out_size, 1);
for i = 1:train_size
    x = (1-a) * x + a * tanh(w * x + w_back * o + bias);
    o = d(:, i);
    % In here the u(i+1) is aligned with x(i+1)
    if i > washout_size
        m(i - washout_size, :) = x';
    end
end

% Compute output weights
w_out = (inv((m' * m) + reg .* eye(res_size)) ...
    * m' * t);

% Compute mean_abs and mse_train
mean_abs = mean(mean(abs(w_out)));
mse_train = mse(w_out' * m', t');

% Each output sequence represents one line of y
y = zeros(out_size,test_size);
internal_testing = zeros(test_size, res_size);
for i = 1:test_size
    x = (1-a) * x + a * tanh(w * x + w_back * o + bias);
    internal_testing(i,:) = x';
    y(:,i) = w_out' * x;
    o = y(:,i);
end

figure();
subplot(4,1,1);
plot(m(:,1:600));
subplot(4,1,2);
plot(internal_testing(:,1:600));
subplot(4,1,3);
plot(1:test_size, y(1,:), 'c');
xlabel('Time[time units]');
ylabel('Signal[signal units]');
subplot(4,1,4);
plot(d(1,:), 'r--');
