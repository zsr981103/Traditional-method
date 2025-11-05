clc;
clear all;
close all;


%%
%数据载入
% origSignal = load('part4.mat');
origSignal = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\part1_4_npy_mat\2007BP_part1_11shot.mat');
origSignal = double(origSignal.data);
% noise = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\mat\snr_6.mat');
% noise = load('Land_2_6_shot.mat');
noise = load('D:\Deep\FFTUNet_Project\本文方法\data\noise_mat_npy_data\part1_4_npy_mat\snr_5.965954873508071.mat');
noise = double(noise.noise_data);
[m n]=size(noise);
% origin = double(origSignal.data);
% [m n]=size(origin);
% 
% noise=[];
% for j=1:n
%     noise0 = noisegen(origin(:,j),5);
%     noise = [noise,noise0];
% end
% %noise = origin + 3*randn(size(origin));
% 
% origSignal=origin;
% errorSignal=origin-noise;
% signal_2 = (sum(origSignal(:).^2));
% noise_2 = (sum(errorSignal(:).^2));
% SNRValues1 = 10*log10(signal_2./noise_2)


%% Step 4: 小波去噪
scale=7;%7
y=[];
tic;
for i=1:n
    x=noise(:,i);
    [c,l]=wavedec(x,scale,'db4');
    [thr,sorh,keepapp]=ddencmp('den','wv',x);
    denoise=wdencmp('gbl',c,l,'db4',scale,thr,sorh,keepapp);
    y=[y,denoise];
end
elapsed_time = toc;
fprintf('去噪所用时间: %.4f 秒\n', elapsed_time);
% save('Wavelet_Sea_2_10_shot_denoise.mat', 'y');
save('Wavelet_denoise.mat', 'y');
%% Step 9: SNR & RMSE 计算
origin = origSignal;
signal_power = sum(origin(:).^2);  % 原始信号的能量

% 计算噪声信号与原始信号的误差能量（MSE的总和）
mse_before_sum = sum((noise(:) - origin(:)).^2);
mse_after_sum = sum((y(:) - origin(:)).^2);

% 计算均方误差 MSE（平均）
mse_before = mse_before_sum / numel(origin);
mse_after = mse_after_sum / numel(origin);

% 计算 SNR
snr_before = 10 * log10(signal_power / mse_before_sum);
snr_after = 10 * log10(signal_power / mse_after_sum);

% 计算 RMSE
rmse_before = sqrt(mse_before);
rmse_after = sqrt(mse_after);

% 计算最大信号幅度（峰值）
max_signal = max(origin(:));

% 计算 PSNR
psnr_before = 10 * log10(max_signal^2 / mse_before);
psnr_after = 10 * log10(max_signal^2 / mse_after);

% 计算 data_range 用于 SSIM
data_range = max(origin(:)) - min(origin(:));

% 计算 SSIM（默认用于二维图像矩阵）
ssim_before = ssim(double(noise), double(origin), 'DynamicRange', data_range);
ssim_after = ssim(double(y), double(origin), 'DynamicRange', data_range);

% 打印结果
fprintf('SNR before: %.2f dB, RMSE before: %.4f, PSNR before: %.2f dB, SSIM before: %.4f\n', ...
    snr_before, rmse_before, psnr_before, ssim_before);
fprintf('SNR after : %.2f dB, RMSE after : %.4f, PSNR after : %.2f dB, SSIM after : %.4f\n', ...
    snr_after, rmse_after, psnr_after, ssim_after);

% 获取当前 GPU 设备信息
g = gpuDevice();

% 记录执行前已分配显存（字节）
beforeMem = g.TotalMemory - g.AvailableMemory;

% 显示执行前显存情况
fprintf('Allocated memory before: %.2f MB\n', beforeMem / 1024^2);

% 这里写你要执行的GPU相关代码，比如分配显存的操作
A = gpuArray.rand(1000, 1000);  % 示例：分配一个1000x1000的GPU数组

% 再次获取 GPU 设备信息，刷新状态
g = gpuDevice();

% 记录执行后已分配显存（字节）
afterMem = g.TotalMemory - g.AvailableMemory;

% 显示执行后显存情况
fprintf('Allocated memory after: %.2f MB\n', afterMem / 1024^2);

% 显示显存增加量
fprintf('Memory increased: %.2f MB\n', (afterMem - beforeMem) / 1024^2);

%%
% b_value = num2str(round(snr_before, 2), '%.2f'); % 保留两位小数
% a_value = num2str(round(snr_after, 2), '%.2f');  % 保留两位小数
% file_name = sprintf('Wavelet_denoise_b%s_a%s.mat', b_value, a_value);

%% Step 9: 绘制剖面图（震道-时间采样点）
% figure;
% vmin = -1;
% vmax = 1;
% % 原始信号
% subplot(2,2,1)
% imagesc(origSignal); colormap(gray); caxis([vmin vmax]); colorbar;
% title('原始信号');
% xlabel('震道');
% ylabel('时间采样点');
% set(gca, 'YDir', 'reverse'); % 让时间向下增加
% 
% 
% % 含噪的信号
% subplot(2,2,2)
% imagesc(noise); colormap(gray); caxis([vmin vmax]); colorbar;
% title('含噪的信号');
% xlabel('震道');
% ylabel('时间采样点');
% set(gca, 'YDir', 'reverse');
% % 
% % % 去噪后的信号
% subplot(2,2,3)
% imagesc(y); colormap(gray); caxis([vmin vmax]); colorbar;
% title('去噪后的信号');
% xlabel('震道');
% ylabel('时间采样点');
% set(gca, 'YDir', 'reverse');
% 
% % 去掉的噪声
% subplot(2,2,4)
% imagesc(y - denoise); colormap(gray); caxis([vmin vmax]); colorbar;
% title('去掉的噪声');
% xlabel('震道');
% ylabel('时间采样点');
% set(gca, 'YDir', 'reverse');
% 
% sgtitle('去噪效果（震道 vs 时间采样点）');
