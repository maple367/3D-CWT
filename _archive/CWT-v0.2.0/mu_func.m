function result = mu_func  (m_iter, n_iter, xi_mat_mr_ns, z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, k0, beta_0, z_lst, eps_lst)
    parfor i = 1:length(m_iter)
        result(i) = mu_func_serial(m_iter(i), n_iter(i), xi_mat_mr_ns(i), z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, k0, beta_0, z_lst, eps_lst);
    end
end

function result = mu_func_serial (m, n, xi_mat_mr_ns, z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, k0, beta_0, z_lst, eps_lst)
    integrand = @(zp, z) xi_mat_mr_ns .* G_func_h(z, zp, m, n, k0, beta_0, z_lst, eps_lst) .* E_profile(zp, z_mesh_raw, E_profile_norm) .* conj(E_profile(z, z_mesh_raw, E_profile_norm));
    mu = k0^2 * integral2(integrand, z_phc_min, z_phc_max, z_phc_min, z_phc_max);
    result = mu;
end
