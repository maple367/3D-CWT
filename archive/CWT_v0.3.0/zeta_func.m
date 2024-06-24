function result = zeta_func (xi_mat_pq, xi_mat_rs, z_mesh_raw, E_profile_norm, z_phc_min, z_phc_max, k0, beta_0, z_lst, eps_lst)
    zeta = -k0^4./(2.*beta_0).*integral2(@(zp, z) xi_mat_pq.*xi_mat_rs.*G_func(z, zp, k0, z_lst, eps_lst).*E_profile(zp, z_mesh_raw, E_profile_norm).*conj(E_profile(z, z_mesh_raw, E_profile_norm)), z_phc_min, z_phc_max, z_phc_min, z_phc_max);
    result = zeta;
end