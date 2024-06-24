function result = n0_func(z, z_lst, eps_lst)
    result = sqrt(interp1(z_lst, eps_lst, z, 'nearest', 'extrap'));
end
