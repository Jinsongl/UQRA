function mcr = get_batch_data(batch_size, ith_batch, filename)
data_src = load(filename).mcr;
mcr = data_src;
idx_begin = batch_size * ith_batch +1;
idx_end   = min(size(data_src.cases,1), batch_size * (ith_batch+1));
mcr.cases= zeros(idx_end - idx_begin+1, 4);
mcr.cases(:, 1:2) = data_src.cases(idx_begin: idx_end, 1:2);
mcr.cases(:, 3) = 1200000;
save('batch_data.mat', 'mcr')