function E_profile = E_profile(z,z_mesh_raw,E_profile_norm)  
    E_profile = interp1(z_mesh_raw, E_profile_norm,z, 'linear', 0.0);  
end  
