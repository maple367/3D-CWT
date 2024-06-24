function G = G_func(z, zp, k0, z_lst, eps_lst)
    function beta_z = beta_z_func(z)
        beta_z = k0 .* n0_func(z, z_lst, eps_lst);
    end
    G = -1j ./ (2 .* beta_z_func(z)) .* exp(-1j .* beta_z_func(z) .* abs(z - zp));
end
