function G_h = G_func_h(z, zp, m, n, k0, beta_0, z_lst, eps_lst)
    function beta_z_h = beta_z_func_h(z, m, n)
        beta_z_h = sqrt((m .^ 2 + n .^ 2) .* beta_0 .^ 2 - k0 .^ 2 .* n0_func(z, z_lst, eps_lst) .^ 2);
    end
    G_h = -1 ./ (2 .* beta_z_func_h(z, m, n)) .* exp(-1 .* beta_z_func_h(z, m, n) .* abs(z - zp));
end
