import qutip as qt


d_A = 2
d_B = 4

ket_AB = qt.rand_ket(dimensions=[d_A, d_B])
print(ket_AB.type, ket_AB.dims, ket_AB.shape)
rho_AB = qt.rand_dm(dimensions=[d_A, d_B])
print(rho_AB.type, rho_AB.dims, rho_AB.shape)
super_AB = qt.rand_super_bcsz(dimensions=[d_A, d_B])
print(super_AB.type, super_AB.dims, super_AB.shape)


super_A_id_B = qt.super_tensor(qt.rand_super_bcsz(d_A), qt.to_super(qt.identity(d_B)))
print(super_A_id_B.type, super_A_id_B.dims, super_A_id_B.shape)

super_of_rho_AB = super_AB(rho_AB)
print(super_of_rho_AB.type, super_of_rho_AB.dims, super_of_rho_AB.shape)

super_A_of_rho_AB = super_A_id_B(rho_AB)
print(super_A_of_rho_AB.type, super_A_of_rho_AB.dims, super_A_of_rho_AB.shape)
