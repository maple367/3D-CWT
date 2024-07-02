#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 28 09:46:16 2024

@author: mestro
"""
from math import *
import numpy as np
import matplotlib.pyplot as plt


L = 1.0
N = 51
dx = L/(N-1)
dy = L/(N-1)
velocity = 2.0
rho = 1
miu = 0.01

P = np.zeros([N,N])
U = np.zeros([N,N+1])
V = np.zeros([N+1,N])
U[N-1,:] = velocity
dU = np.zeros([N,N+1])
dV = np.zeros([N+1,N])


P_iter = P.copy()
U_iter = U.copy()
V_iter = V.copy()

NN = N*N
iter = 0
err = 1.0

while (iter < 1000) and (err > 1e-5):
    for i in range(1,N-1):
        for j in range(1,N):
            rho_u_e = 0.5*(U[i,j] + U[i,j+1])*rho
            rho_u_w = 0.5*(U[i,j] + U[i,j-1])*rho
            rho_v_s = 0.5*(V[i,j] + V[i,j-1])*rho
            rho_v_n = 0.5*(V[i+1,j] + V[i+1,j-1])*rho
            
            AE = 0.5*(abs(rho_u_e) + rho_u_e)*dy + miu*dy/dx
            AW = 0.5*(abs(rho_u_w) - rho_u_w)*dy + miu*dy/dx
            AN = 0.5*(abs(rho_v_n) + rho_v_n)*dx + miu*dx/dy
            AS = 0.5*(abs(rho_v_s) - rho_v_s)*dx + miu*dx/dy
            
            AEE = 0.5*(abs(rho_u_e) - rho_u_e)*dy + miu*dy/dx
            AWW = 0.5*(abs(rho_u_w) + rho_u_w)*dy + miu*dy/dx
            ANN = 0.5*(abs(rho_v_n) - rho_v_n)*dx + miu*dx/dy
            ASS = 0.5*(abs(rho_v_s) + rho_v_s)*dx + miu*dx/dy
            
            Ap = (AE+AW+AN+AS)
            U_iter[i,j] = 1/Ap*(AEE*U[i,j+1] + AWW*U[i,j-1] + ANN*U[i+1,j] + ASS*(U[i-1,j]) - (P[i,j] - P[i,j-1])*dy)
            dU[i,j] = dy/Ap
    #bottom mesh
    i = 0
    for j in range(1,N):
        rho_u_e = 0.5*(U[i,j] + U[i,j+1])*rho
        rho_u_w = 0.5*(U[i,j] + U[i,j-1])*rho
        #rho_v_s = 0.5*(V[i,j] + V[i,j-1])*rho
        rho_v_n = 0.5*(V[i+1,j] + V[i+1,j-1])*rho
        
        AE = 0.5*(abs(rho_u_e) + rho_u_e)*dy + miu*dy/dx
        AW = 0.5*(abs(rho_u_w) - rho_u_w)*dy + miu*dy/dx
        AN = 0.5*(abs(rho_v_n) + rho_v_n)*dx + miu*dx/dy
        #AS = 0.5*(abs(rho_v_s) - rho_v_s)*dx + miu*dx/dy
        As = 0
        Ap = (AE+AW+AN+AS)
        dU[i,j] = dy/Ap
    #top mesh
    i = N-1
    for j in range(1,N):
        rho_u_e = 0.5*(U[i,j] + U[i,j+1])*rho
        rho_u_w = 0.5*(U[i,j] + U[i,j-1])*rho
        rho_v_s = 0.5*(V[i,j] + V[i,j-1])*rho
        #rho_v_n = 0.5*(V[i+1,j] + V[i+1,j-1])*rho
        
        AE = 0.5*(abs(rho_u_e) + rho_u_e)*dy + miu*dy/dx
        AW = 0.5*(abs(rho_u_w) - rho_u_w)*dy + miu*dy/dx
        #AN = 0.5*(abs(rho_v_n) + rho_v_n)*dx + miu*dx/dy
        AN = 0.0
        AS = 0.5*(abs(rho_v_s) - rho_v_s)*dx + miu*dx/dy
        Ap = (AE+AW+AN+AS)
        dU[i,j] = dy/Ap
    #Apple BCs
    U_iter[:,0] = -U_iter[:,1] #left
    U_iter[1:N-1,N] = -U_iter[1:N-1,N-1] #right
    U_iter[0,:] = 0.0 #bottom
    U_iter[N-1,:] = velocity #top
    
    
    # V equation
    for i in range(1,N):
        for j in range(1,N-1):
            rho_u_e = 0.5*(U[i,j] + U[i-1,j])*rho
            rho_u_w = 0.5*(U[i,j+1] + U[i-1,j+1])*rho
            rho_v_n = 0.5*(V[i,j] + V[i+1,j])*rho
            rho_v_s = 0.5*(V[i,j] + V[i-1,j])*rho
            
            AE = 0.5*(abs(rho_u_e) + rho_u_e)*dy + miu*dy/dx
            AW = 0.5*(abs(rho_u_w) - rho_u_w)*dy + miu*dy/dx
            AN = 0.5*(abs(rho_v_n) + rho_v_n)*dx + miu*dx/dy
            AS = 0.5*(abs(rho_v_s) - rho_v_s)*dx + miu*dx/dy
            
            AEE = 0.5*(abs(rho_u_e) - rho_u_e)*dy + miu*dy/dx
            AWW = 0.5*(abs(rho_u_w) + rho_u_w)*dy + miu*dy/dx
            ANN = 0.5*(abs(rho_v_n) - rho_v_n)*dx + miu*dx/dy
            ASS = 0.5*(abs(rho_v_s) + rho_v_s)*dx + miu*dx/dy
            
            Ap = (AE+AW+AN+AS)
            V_iter[i,j] = 1.0/Ap*(AEE*V[i,j+1] + AWW*V[i,j-1] + ANN*V[i+1,j] + ASS*V[i-1,j] - (P[i,j] - P[i-1,j])*dx)
            dV[i,j] = dx/Ap
    #left
    j = 0
    for i in range(1,N):
        rho_u_e = 0.5*(U[i,j] + U[i-1,j])*rho
        #rho_u_w = 0.5*(U[i,j+1] + U[i-1,j+1])*rho
        rho_v_n = 0.5*(V[i,j] + V[i+1,j])*rho
        rho_v_s = 0.5*(V[i,j] + V[i-1,j])*rho
        
        AE = 0.5*(abs(rho_u_e) + rho_u_e)*dy + miu*dy/dx
        #AW = 0.5*(abs(rho_u_w) - rho_u_w)*dy + miu*dy/dx
        AW = 0.0
        AN = 0.5*(abs(rho_v_n) + rho_v_n)*dx + miu*dx/dy
        AS = 0.5*(abs(rho_v_s) - rho_v_s)*dx + miu*dx/dy
        Ap = (AE+AW+AN+AS)
        dV[i,j] = dx/Ap
    
    #right0, L, N
    j = N-1
    for i in range(1,N):
        #rho_u_e = 0.5*(U[i,j] + U[i-1,j])*rho
        rho_u_w = 0.5*(U[i,j+1] + U[i-1,j+1])*rho
        rho_v_n = 0.5*(V[i,j] + V[i+1,j])*rho
        rho_v_s = 0.5*(V[i,j] + V[i-1,j])*rho
        
        #AE = 0.5*(abs(rho_u_e) + rho_u_e)*dy + miu*dy/dx
        AE = 0.0
        AW = 0.5*(abs(rho_u_w) - rho_u_w)*dy + miu*dy/dx
        AN = 0.5*(abs(rho_v_n) + rho_v_n)*dx + miu*dx/dy
        AS = 0.5*(abs(rho_v_s) - rho_v_s)*dx + miu*dx/dy
        Ap = (AE+AW+AN+AS)
        dV[i,j] = dx/Ap
    #Apply BCs

    V_iter[:,0] = 0.0
    V_iter[:,N-1] = 0.0
    V_iter[0,1:N-1] = -V_iter[1,1:N-1]
    V_iter[N,1:N-1] = -V_iter[N-1,1:N-1]
    
    U_old = U.copy()
    V_old = V.copy()
    bp = np.zeros([NN,1])
    #pressure fix
    for i in range(N):
        for j in range(N):
            index = i*N+j
            bp[index] = (rho*U_iter[i,j]*dy - rho*U_iter[i,j+1]*dy + rho*V_iter[i,j]*dx - rho*V_iter[i+1,j]*dx)
    bp[0] = 0.0
    
    APP = np.zeros([NN,NN])
    #left bottom
    i = 0
    j = 0
    index = i*N + j
    # Ae = -rho*dU[i,j+1]*dy
    # An = -rho*dV[i+1,j]*dx
    # Ap = -(Ae + An)
    # APP[index,index+1] = Ae
    # APP[index,index+N] = An
    # APP[index,index] = Ap
    APP[index,index] = 1
    
    #right bottom
    i = 0
    j = N-1
    index = i*N + j
    Aw = -rho*dU[i,j]*dy
    An = -rho*dV[i+1,j]*dx
    Ap = -(Aw + An)
    APP[index,index - 1] = Aw
    APP[index,index + N] = An
    APP[index,index] = Ap
    
    #left top
    i = N-1
    j = 0
    index = i*N + j
    As = -rho*dV[i,j]*dx
    Ae = -rho*dU[i,j+1]*dy
    Ap = -(As + Ae)
    APP[index,index+1] = Ae
    APP[index,index-N] = As
    APP[index,index] = Ap
    
    #right top
    i = N-1
    j = N-1
    index = i*N+j
    Aw = -rho*dU[i,j]*dy
    As = -rho*dV[i,j]*dx
    Ap = -(Aw + As)
    APP[index,index] = Ap
    APP[index,index-1] = Aw
    APP[index,index-N] = As 

    i = 0
    for j in range(1,N-1):
        index = i*N+j
        Aw = -rho*dU[i,j]*dy
        An = -rho*dV[i+1,j]*dx
        Ae = -rho*dU[i,j+1]*dy
        Ap = -(Aw + An + Ae)
        APP[index,index] = Ap
        APP[index,index-1] = Aw
        APP[index,index+N] = An
        APP[index,index+1] = Ae
        
        
        
    i = N-1
    for j in range(1,N-1):
        index = i*N+j
        Aw = -rho*dU[i,j]*dy
        As = -rho*dV[i,j]*dx
        Ae = -rho*dU[i,j+1]*dy
        Ap = -(Aw + As + Ae)
        APP[index,index] = Ap
        APP[index,index - 1] = Aw
        APP[index,index - N] = As
        APP[index,index + 1] = Ae
    
    j = 0
    for i in range(1,N-1):
        index = i*N+j
        Ae = -rho*dU[i,j+1]*dy
        An = -rho*dV[i+1,j]*dx
        As = -rho*dV[i,j]*dx
        Ap = -(Ae + An + As)
        APP[index,index] = Ap
        APP[index,index + 1] = Ae
        APP[index,index + N] = An
        APP[index,index - N] = As
    
    j = N-1
    for i in range(1,N-1):
        index = i*N + j
        Aw = -rho*dU[i,j]*dy
        An = -rho*dV[i+1,j]*dx
        As = -rho*dV[i,j]*dx
        Ap = -(Aw + An + As)
        APP[index,index] = Ap
        APP[index,index - 1] = Aw
        APP[index,index + N] = An
        APP[index,index - N] = As
    
    for i in range(1,N-1):
        for j in range(1,N-1):
            index = i*N + j
            Aw = -rho*dU[i,j]*dy
            An = -rho*dV[i+1,j]*dx
            As = -rho*dV[i,j]*dx
            Ae = -rho*dU[i,j+1]*dy
            Ap = -(Aw + An + As + Ae)
            APP[index,index] = Ap
            APP[index,index - 1] = Aw
            APP[index,index + N] = An
            APP[index,index - N] = As
            APP[index,index + 1] = Ae
            
            
    # pressure correction
    p_fix = np.linalg.solve(APP, bp)
    P_fix_matrix = np.zeros([N,N])
    for i in range(N):
        for j in range(N):
            index = i*N+j
            P[i,j] = 0.3*p_fix[index] + P[i,j]
            P_fix_matrix[i,j] = p_fix[index]
    P[0,0] = 0.0
    #velocity update
    for i in range(1,N-1):
        for j in range(1,N):
            U[i,j] = U_iter[i,j] + dU[i,j]*(P_fix_matrix[i,j-1] - P_fix_matrix[i,j])
    for i in range(1,N):
        for j in range(1,N-1):
            V[i,j] = V_iter[i,j] + dV[i,j]*(P_fix_matrix[i-1,j] - P_fix_matrix[i,j])
    
    
    #Apple BCs
    U[1:N-1,0] = -U[1:N-1,1] #left
    U[1:N-1,N] = -U[1:N-1,N-1] #right
    U[0,:] = 0.0 #bottom
    U[N-1,:] = velocity #top
    
    V[0,1:N-1] = -V[1,1:N-1]
    V[N,1:N-1] = -V[N-1,1:N-1]
    V[:,0] = 0.0
    V[:,N-1] = 0.0
    
    err1 = np.max(np.abs(U-U_old))
    err2 = np.max(np.abs(V-V_old))
    err = max(err1,err2)
    
    iter = iter + 1
    print("\rthe iter num is " + str(iter) + " and max err is " + str(err) + "          ", end="", flush=True)
    


#plot
x = np.linspace(dx/2, 1 - dx/2,N-1)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)  # Generate 2D grid coordinates
plt.figure()
plt.contourf(X,Y, U[:,1:N], levels=20, cmap='jet')  # Adjust levels and cmap as needed
plt.colorbar(label='Velocity UX')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Velocity Contour Plot')
plt.show()


x = np.linspace(0,1,N)
y = np.linspace(0, 1, N)
X, Y = np.meshgrid(x, y)  # Generate 2D grid coordinates
plt.figure()
plt.contourf(X,Y,P, levels=20, cmap='jet')  # Adjust levels and cmap as needed
plt.colorbar(label='pressure P')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('Pressure')
plt.show()
