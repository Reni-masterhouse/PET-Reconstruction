%%
clear;

%确定分辨率
res = [140, 140, 120]; 

%输出图像时进行center crop
crop_rate = 0.1; 

%确定视野范围(mm)
fov_size = [70, 70, 96]; 

grid_size = fov_size ./ res; 

name = 'HotRod39M';
%name = '2PointSource';
fid = fopen(['LOR', name, '_clr'], 'r');
bin_data = fread(fid, 'float=>double');
fclose(fid);

data = reshape(bin_data, 7, [])';
data = data(data(:, 7) > 0, :); 

%% 
%确定subset大小
batch_size = 2^18; 

batch_num = ceil(size(data, 1) / batch_size);

%初始化
fi = ones([1, prod(res)]); 
max_iter = 2;

%记录投影误差
errs = zeros(max_iter, batch_num); 

for iter = 1:max_iter
    
    %随机LOR顺序
    data = data(randperm(size(data,1)),:);
    
    %编写waitbar
    wb = waitbar(0, 'process');
    
    for batch_idx = 1:batch_num
        
        %拿出对应子集
        batch_data = data(batch_size*(batch_idx-1)+1:min(batch_size*batch_idx, size(data,1)),:);
        x1 = batch_data(:, 1:3) ./ grid_size;
        x2 = batch_data(:, 4:6) ./ grid_size;
        pj = batch_data(:, 7);

        bs = size(batch_data, 1);
        
        %稀疏化的传输矩阵
        cij_x = zeros(bs*sum(res),1,'int32');
        cij_y = zeros(bs*sum(res),1,'int32');
        cij_v = zeros(bs*sum(res),1);
        num_cij = 0;
        
        %线模型siddon算法
        x_plane_1 = -0.5 * res;
        x_plane_N = 0.5 * res;

        alpha_1 = max(min((x_plane_1 - x1) ./ (x2 - x1), 1),0);
        alpha_N = max(min((x_plane_N - x1) ./ (x2 - x1), 1),0);

        alpha_min = max(min(alpha_1, alpha_N), [], 2);
        alpha_max = min(max(alpha_1, alpha_N), [], 2);
        dx = x2 - x1;

        ijk_cond1 = x1 + alpha_min .* dx - x_plane_1;
        ijk_cond2 = x1 + alpha_max .* dx - x_plane_1;
        ijk_min = zeros(size(ijk_cond1), 'like', ijk_cond1);
        cond = dx > 0;
        ijk_min(cond) = ijk_cond1(cond);
        ijk_min(~cond) = ijk_cond2(~cond);

        ijk_max = zeros(size(ijk_cond1), 'like', ijk_cond1);
        ijk_max(cond) = ijk_cond2(cond);
        ijk_max(~cond) = ijk_cond1(~cond);
        has_intersect = alpha_min < alpha_max;
        assert(all(ijk_max(has_intersect) - ijk_min(has_intersect) >=0));
        ijk_min = int32(ceil(ijk_min));
        ijk_max = int32(floor(ijk_max));
        alphas = [alpha_min, alpha_max];
        d = vecnorm(dx, 2, 2);
        
        for ii = 1:bs
            if ~has_intersect(ii)
                continue
            end
            
            %计算所有的alpha
            new_as = alphas(ii, :);
            for j = 1:3
                if dx(ii, j) ~= 0
                    new_set = (double(ijk_min(ii, j):ijk_max(ii,j)) + x_plane_1(j) - x1(ii, j)) ./ dx(ii, j);
                    new_as = [new_as, new_set];
                end
            end
            new_as = unique(new_as);
            l = length(new_as);
            line_len = (new_as(2:l) - new_as(1:l-1)) .* d(ii); % 相交线的长度
            mid_pos = (new_as(2:l) + new_as(1:l-1))' / 2 .* dx(ii, :) + x1(ii, :); 
            mid_idx = int32(floor(mid_pos - x_plane_1)) + 1; % 确定相交线所处的体素
            valid_line = line_len > 1e-4;
            f_idx = sub2ind(res, mid_idx(valid_line, 1), mid_idx(valid_line, 2), mid_idx(valid_line, 3));

            %更新传输矩阵
            valid_num = sum(valid_line);
            cij_x(num_cij + 1:num_cij + valid_num) = f_idx;
            cij_y(num_cij + 1:num_cij + valid_num) = ii;
            cij_v(num_cij + 1:num_cij + valid_num) = line_len(valid_line);
            num_cij = num_cij + valid_num;
        end
        
        %建立传输矩阵
        cij = sparse(cij_x(1:num_cij), cij_y(1:num_cij), cij_v(1:num_cij), numel(fi), bs);
        proj_this = fi * cij;
        
        %排除没有和视野相交的LOR
        mask_j = proj_this ~= 0;
        cij_sum = sum(cij, 2)';
        
        %排除没有对LOR有贡献的体素
        mask_i = cij_sum > 0;
        cij_masked = cij(mask_i, mask_j);
        
        %迭代更新
        ratio = transpose(pj(mask_j)) ./ (fi(mask_i) * cij_masked);
        eff = (cij_masked * transpose(ratio))' ./ cij_sum(mask_i);
        fi(mask_i) = fi(mask_i) .* eff;
        err = sum(abs(transpose(pj) - proj_this))/ sum(mask_j);
        errs(iter,batch_idx) = err;
        avg_err = sum(errs(iter, 1:batch_idx)) / batch_idx;
        str=['iter ', num2str(iter), ' Processing...',num2str(batch_idx/batch_num*100),'%,', ' avg err ', num2str(avg_err), ' err ', num2str(err)];
        waitbar(batch_idx/batch_num, wb, str)
        
        %保存轴向的热源强度之和
        img = sum(reshape(fi, res), 3);
        crop_img = img(res(1)*crop_rate:res(1)*(1-crop_rate), res(2)*crop_rate:res(2)*(1-crop_rate), :);
        crop_img = crop_img / max(crop_img, [], 'all');
        path = [name, '/', num2str(iter), '_', num2str(batch_idx), '.png'];
        imwrite(crop_img, path);
    end
    delete(wb);
end

%%
%保存体数据
vol = reshape(fi, res);
fid = fopen([name, '/vol_hot.dat'], 'wb');
fwrite(fid, vol, 'single');
fclose(fid);

%%
%保存轴向截面图像
f_vol = reshape(fi, res);
fmax = max(f_vol, [], 'all');
for i=1:size(f_vol,3)
    imwrite(f_vol(:, :, i), [name, '/vol_',num2str(i),'.png']) 
end

