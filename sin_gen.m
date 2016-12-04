clear all;
clc;

% Input, reservoir, output size
res_size = 20;
in_size = 0;
out_size = 1;

% Training data size and washout time
test_size = 50;
train_size = 300;
washout_size = 100;

% Scaling parameters
w_scale = 0.98;
w_back_scale = 1.01;
bias_scale = 0;
reg = 1e-8;

rand('seed', 42);
% Scale internal weights w
w_0 = 2 * rand(res_size, res_size) - 1;
spectral_rad_w_0 = max(abs(eigs(w_0)));
w = w_0 * ((1.00 / spectral_rad_w_0) * w_scale);

% Compute w_back
w_back = (2 * rand(res_size, out_size) - 1) * w_back_scale;

% Compute bias
bias = (2 * rand(res_size, out_size) - 1) * bias_scale;

% Compute teacher forcing vector
sample_points = 1:(train_size + test_size + 1);
d = sin(2 * pi * sample_points/20);

% Internal state collector m and teacher collector t
m = zeros(train_size - washout_size, res_size);
t_T = transpose(d);
t = t_T(washout_size + 1:train_size,:);

x = zeros(res_size, 1);
u = 0;
for i = 1:train_size
    x = tanh(w * x + w_back * u + bias);
    u = d(i);
    if i > washout_size
        m(i - washout_size, :) = transpose(x);
    end
end

% Compute output weights
w_out = (inv((transpose(m) * m) + reg .* eye(res_size)) * transpose(m) * t);

% Compute mean_abs and mse_train
mean_abs = mean(abs(w_out));
mse_train = mse(w_out' * m', t');

y = zeros(1,test_size);
internal_testing = zeros(test_size, res_size);
for i = 1:test_size
    x = tanh(w * x + w_back * u + bias);
    internal_testing(i,:) = transpose(x);
    y(i) = transpose(w_out) * x;
    u = y(i);
end

% Compute mse_test
mse_test = mse(y, d(train_size + 1:train_size + test_size));

figure;
% subplot(3,1,1);
% plot(m(:,1:5));
% subplot(3,1,2);
% plot(internal_testing(:,1:5));
% subplot(3,1,3);
p = plot(1:test_size, y, 'c', 1:test_size, d(train_size + 1:train_size + test_size), 'r--');
p(1).LineWidth = 4;