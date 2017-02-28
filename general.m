clear all;
clc;

rand('seed', 14);

% Input, reservoir, output size
res_size = 100;
in_size = 3;
out_size = 3;

% Training data size and washout time
test_size = 500;
train_size = 300;
washout_size = 100;

% Scaling parameters
w_scale = 1;
w_back_scale = 1;
w_in_scale = 1;
bias_scale = 0;
reg = 1e-8;

% Leaking rate
a = 0.8;

% Scale internal weights w
w_0 = 2 * rand(res_size, res_size) - 1;
spectral_rad_w_0 = max(abs(eigs(w_0)));
w = w_0 * ((1.00 / spectral_rad_w_0) * w_scale);

% Compute w_back
w_back = (2 * rand(res_size, out_size) - 1) * w_back_scale;

% Compute w_in (in_size + 1 to have a 1 in the system at all times
w_in = (2 * rand(res_size, in_size + 1) - 1) * w_in_scale;

% Compute bias
bias = (2 * rand(res_size, 1) - 1) * bias_scale;

% Compute teacher forcing vector
sample_points = 1:(train_size + test_size + 1);
d1 = sin(2 * pi * sample_points/40);
d2 = cos(2 * pi * sample_points/10);
d3 = sawtooth(2 * pi * sample_points/30);
d = vertcat(d1, d2, d3);

% Internal state collector m and teacher collector t
m = zeros(train_size - washout_size, res_size + in_size + 1);
t_T = d';
t = t_T(washout_size + 1:train_size,:);

% x column vector
x = zeros(res_size, 1);
o = zeros(out_size, 1);
for i = 1:train_size
    u = d(:, i);
    x = (1-a) * x + a * tanh(w * x + w_in * vertcat(1, u) + w_back * o + bias);
    o = u;
    % In here the u(i+1) is aligned with x(i+1)
    if i > washout_size
        m(i - washout_size, :) = horzcat(1, u', x');
    end
end

% Compute output weights
w_out = (inv((m' * m) + reg .* eye(res_size + in_size + 1)) ...
    * m' * t);

% Compute mean_abs and mse_train
mean_abs = mean(abs(w_out));
mse_train = mse(w_out' * m', t');

% Each output sequence represents one line of y
y = zeros(out_size,test_size);
internal_testing = zeros(test_size, res_size);
for i = 1:test_size
    u = d(:, train_size + i);
    x = (1-a) * x + a * tanh(w * x + w_in * vertcat(1,u) + w_back * o + bias);
    internal_testing(i,:) = x';
    y(:,i) = w_out' * vertcat(1, u, x);
    o = y(:,i);
end

% Compute mse_test
mse_test = mse(y, d(train_size + 1:train_size + test_size));

figure();
subplot(3,1,1);
plot(m(:,1:5));
subplot(3,1,2);
plot(internal_testing(:,1:5));
subplot(3,1,3);
p = plot(1:test_size, y(1,:), 'c', 1:test_size, d(1, train_size + 1:train_size + test_size), 'r--');
p(1).LineWidth = 4;
xlabel('Time[time units]');
ylabel('Signal[signal units]');
legend('show');
legend('output', 'teacher');