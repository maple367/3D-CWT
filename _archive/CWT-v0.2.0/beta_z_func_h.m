function result = beta_z_func_h(z, m, n,k0,beta_0,z_lst,eps_lst)
    result = sqrt((m^2 + n^2) * beta_0^2 - k0^2 * n0_func(z,z_lst,eps_lst).^2);
end
