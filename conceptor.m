clear all;
clc;

rng('default');
% Input, reservoir, output size
res_size = 200;

% Training data size and washout time
test_pattern_size = 100;
test_morph_size = 50;
train_size = 600;
washout_size = 100;
eff_train_size = train_size - washout_size;

% Scaling parameters
w_star_scale = 1.6;
w_in_scale = 1.6;
bias_scale = 0.3;
reg = 1e-8;

% Apertures for the patterns. Must match number of patterns
aperture = [10 10 5 1 10];

% Compute w_star
w_0 = randn(res_size, res_size);
% TODO: Change if things don't work
spectral_rad_w_0 = max(abs(eigs(w_0)));
w_star = w_0 * (1.00 / spectral_rad_w_0) * w_star_scale;

% Compute w_in
w_in = randn(res_size, 1) * w_in_scale;

% Compute bias
bias = randn(res_size, 1) * bias_scale;

% Patterns (More that 1 pattern is required for the script to work)
period = 10;

% - Pattern 1
pattern{1} = @(i) sin(2 * pi * i / period);

% - Pattern 2
pattern{2} = @(i) sawtooth(2 * pi * i / period);

% - Pattern 3
pattern{3} = @(i) 0;

% - Pattern 4
pattern{4} = @(i) (cos(2 * pi * i / period));

% This allows to dynamically increase the number of patterns
pattern_cell_size = size(pattern);
pattern_number = pattern_cell_size(2);

for i = 1:pattern_number
    current_pattern = pattern{i};
    output_aligned_m = zeros(res_size, eff_train_size);
    input_aligned_m = zeros(res_size, eff_train_size);
    single_pattern_output = zeros(1, eff_train_size);
    x = zeros(res_size, 1);
    % x(n+1) = tanh(w_star * x(n) + w_in * p(n) + bias);
    % y(n) = p(n)
    for j = 1:train_size
        x_old = x;
        u = current_pattern(j);
        x = tanh(w_star * x + w_in * u + bias);
        if j > washout_size
            output_aligned_m(:,j - washout_size) = x;
            input_aligned_m(:,j - washout_size) = x_old;
            single_pattern_output(1, j - washout_size) = u;
        end;   
    end;
    R{i} = output_aligned_m * output_aligned_m' / eff_train_size;
    
    output_aligned_state(:, (i - 1) * eff_train_size + 1: i * eff_train_size) = ...
        output_aligned_m;
    input_aligned_state(:, (i - 1) * eff_train_size + 1: i * eff_train_size) = ...
        input_aligned_m;
    output(1, (i - 1) * eff_train_size + 1: i * eff_train_size) = ...
        single_pattern_output;
end;

% Compute w_out
% In this case the output y(n+1) = p(n) will be aligned with x(n+1)
normalize_output = pattern_number * eff_train_size;
% Minimizes sum(j){sum(n)|y(n) - w_out * x(n)|^2}
w_out = (inv(output_aligned_state * output_aligned_state' / normalize_output ...
    + reg * eye(res_size)) * output_aligned_state * output' / normalize_output)';
% Compute training error
train_error = mse(w_out * output_aligned_state, output);

% Compute w
% In this case the state x(n) = x_old will be aligned with x(n)
no_bias_state = (atanh(output_aligned_state) - repmat(bias, 1, pattern_number * eff_train_size));
% Minimizes sum(j){sum(n)|w_star * x(n) + w_in * p(n) - w * x(n)|^2}
w = (inv(input_aligned_state * input_aligned_state' / normalize_output ...
    + reg * eye(res_size)) * input_aligned_state * no_bias_state' / normalize_output)';
w_error = mse(w * input_aligned_state, no_bias_state);

% Compute Conceptors
C = cell(1, pattern_number);
% TODO: Use R_i directly
for i = 1:pattern_number
    [U S V] = svd(R{i});
    S_n = (S * inv(S + aperture(i)^(-2) * eye(res_size)));
    C{1, i} = U * S_n * U';
end;

% Allocate 1 cell for each pattern + 1 cell for each morphing between 2
% consequtive patterns
plot_data_size = 2 * pattern_number - 1;
plot_data = cell(1, plot_data_size);

for i = 1:plot_data_size
    int_division = fix(i/2);
    if mod(i, 2) == 0
        morph_output = zeros(test_morph_size, 1);
        C_1 = C{1, int_division};
        C_2 = C{1, int_division + 1};
        for j = 1:test_morph_size
            coeff = j / test_morph_size;
            x = ((1 - coeff) * C_1 + coeff * C_2) * tanh(w * x + bias);
            morph_output(j,1) = w_out * x;
        end
        plot_data{i} = morph_output;
    else
        pattern_output = zeros(test_pattern_size, 1);
        C_1 = C{1, i - int_division};
        for j = 1:test_pattern_size
            x = C_1 * tanh(w * x + bias);
            pattern_output(j,1) = w_out * x;
        end
        plot_data{i} = pattern_output;
    end
end

for i = 1:(pattern_number - 1)
    index = 2 * i - 1;
    figure(i);
    subplot(3,1,1);
    plot(plot_data{1, index}, 'r');
    subplot(3,1,2);
    plot(plot_data{1, index + 2}, 'b');
    subplot(3,1,3);
    all_data = vertcat(plot_data{1,index},plot_data{1, index+1},plot_data{1, index+2});
    plot(all_data, 'cyan');
    hold on;
    plot(1:test_pattern_size, plot_data{1,index}, 'r', ...
        (test_pattern_size + 1):(test_morph_size + test_pattern_size), plot_data{1, index+1}, 'g', ...
        (test_morph_size + test_pattern_size + 1):(test_morph_size + 2 * test_pattern_size), plot_data{1, index+2}, 'b');
    hold off;
end
