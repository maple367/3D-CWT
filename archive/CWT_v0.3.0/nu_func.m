function result = nu_func (xi_mat_mr_ns, z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, z_lst, eps_lst)
    parfor i = 1:length(xi_mat_mr_ns)
        result(i) = nu_func_serial(xi_mat_mr_ns(i), z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, z_lst, eps_lst);
    end
end

function result = nu_func_serial(xi_mat_mr_ns, z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, z_lst, eps_lst)
    integrand = @(z) 1./(n0_func(z,z_lst,eps_lst).^2) .* xi_mat_mr_ns.* abs(E_profile(z,z_mesh_raw,E_profile_norm)).^2;
    nu = -integral(integrand, z_phc_min, z_phc_max);
    result = nu;
end