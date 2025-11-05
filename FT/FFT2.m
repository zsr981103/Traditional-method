
% 傅里叶变换去噪
clc
clear all
close all

%%
%数据载入
%load('matlab.mat')
% origSignal = load('part4.mat');
origSignal = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\part1_4_npy_mat\2007BP_part1_11shot.mat');
origSignal = double(origSignal.data);
% noise = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\mat\snr_6.mat');
noise = load('D:\Deep\FFTUNet_Project\data_and_result\Mobil_Avo_Viking_Graben_Line_12真实海洋波\mat\Sea_0_1_shot.mat');
% noise = load('Land_2_6_shot.mat');
noise = double(noise.data);
cleanSignal = origSignal;
noisySignal = noise;

% 显示原始和带噪声信号
vmin = -1;
vmax = 1;
figure;
subplot(2,1,1); imagesc(cleanSignal); colormap(gray); caxis([vmin vmax]); colorbar; title('原始信号');
subplot(2,1,2); imagesc(noisySignal); colormap(gray); caxis([vmin vmax]); colorbar; title('带噪声信号');
tic;
% 3. 计算二维傅里叶变换
F = fft2(noisySignal);
F_shifted = fftshift(F);  % 将零频率移到中心

% 显示频谱
magnitude = log(1 + abs(F_shifted));
figure; imagesc(magnitude); colorbar; title('傅里叶频谱');
colormap('jet');

% 4. 设计频域滤波器
[M, N] = size(noisySignal);
D0 = 300;  % 截止频率（可根据需要调整）

% 创建网格坐标
u = -floor(M/2):ceil(M/2)-1;
v = -floor(N/2):ceil(N/2)-1;
[V, U] = meshgrid(v, u);
D = sqrt(U.^2 + V.^2);

% 创建高斯低通滤波器
H = exp(-(D.^2)./(2*D0^2));

% 显示滤波器
figure; imagesc(H); colorbar; title('频域滤波器');
colormap('jet');

% 5. 应用滤波器
G_shifted = H .* F_shifted;
G = ifftshift(G_shifted);  % 移回原始位置
filteredSignal_gauss = real(ifft2(G));

% 6. 逆傅里叶变换
filteredSignal = real(ifft2(G));

elapsed_time = toc;
fprintf('去噪所用时间: %.4f 秒\n', elapsed_time);
y = filteredSignal;
save('FFT_denoise.mat', 'y');
% 7. 显示结果
figure;
subplot(3,1,1); imagesc(cleanSignal); colormap(gray); caxis([vmin vmax]); colorbar;  title('原始信号');
subplot(3,1,2); imagesc(noisySignal); colormap(gray); caxis([vmin vmax]); colorbar;  title('带噪声信号');
subplot(3,1,3); imagesc(filteredSignal); colormap(gray); caxis([vmin vmax]); colorbar;  title('去噪后信号');

% 8. 计算信噪比评估去噪效果
% originalPower = sum(cleanSignal(:).^2);
% noisePower = sum((noisySignal(:)-cleanSignal(:)).^2);
% filteredNoisePower = sum((filteredSignal(:)-cleanSignal(:)).^2);
% 
% SNR_original = 10*log10(originalPower/noisePower);
% SNR_filtered = 10*log10(originalPower/filteredNoisePower);
% rmse_before = sqrt(noisePower / numel(cleanSignal));
% rmse_after  = sqrt(filteredNoisePower / numel(cleanSignal));
% fprintf('SNR before: %.2f dB, RMSE before: %.4f\n', SNR_original, rmse_before);
% fprintf('SNR after : %.2f dB, RMSE after : %.4f\n', SNR_filtered, rmse_after);
% fprintf('原始信噪比: %.2f dB\n', SNR_original);
% fprintf('去噪后信噪比: %.2f dB\n', SNR_filtered);

% 9. 可选：查看某一行或列的时域比较
% rowToShow = 500;  % 选择显示第500行
% figure;
% plot(cleanSignal(:,rowToShow), 'b', 'LineWidth', 1.5); hold on;
% plot(noisySignal(:,rowToShow), 'r', 'LineWidth', 0.5);
% plot(filteredSignal(:,rowToShow), 'g', 'LineWidth', 1.5);
% legend('原始信号', '带噪声信号', '去噪后信号');
% title(['第', num2str(rowToShow), '行信号比较']);
% xlabel('样本点'); ylabel('幅值');