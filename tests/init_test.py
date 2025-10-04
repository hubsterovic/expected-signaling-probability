import qutip as qt
import pytest


@pytest.fixture(
    params=[
        ("small_sym", 2, 2),
        ("small_asym", 2, 4),
        ("medium_sym", 4, 4),
        ("medium_asym", 3, 5),
        ("large_sym", 8, 8),
        ("large_asym", 4, 16),
    ]
)
def dims(request):
    return request.param


def test_superoperator_supertensor_identity(dims):
    _, d_A, d_B = dims
    rho_AB = qt.rand_dm(dimensions=[d_A, d_B])
    super_A_id_B = qt.super_tensor(qt.rand_super_bcsz(d_A), qt.to_super(qt.identity(d_B)))
    assert super_A_id_B.type == "super"

    rho_B_original = qt.ptrace(rho_AB, 1)
    super_A_id_B_of_rho_AB = super_A_id_B(rho_AB)
    rho_B_after_super_A = qt.ptrace(super_A_id_B_of_rho_AB, 1)

    difference = rho_B_original - rho_B_after_super_A
    assert difference.norm(norm="tr") < 1e-10
