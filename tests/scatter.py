import jax
import jax.numpy as jnp

def scatter_to_grid(full, tensor, maps):
    T, N = maps.shape
    G = full.shape[0]
    middle = tensor.shape[1:-2]
    flat_tensor = tensor.reshape(T, -1, N, N)

    @jax.vmap
    def scatter(t, m):
        i, j = m[:, None], m[None, :]
        return jnp.broadcast_to(full, (t.shape[0], G, G)).copy().at[:, i, j].set(t)

    return scatter(flat_tensor, maps).reshape(T, *middle, G, G)


# --- Test valeurs 1 : cas minimal sans dims intermédiaires ---
# T=2, N=2, G=3, full=zeros
# sample 0 : maps=[0,2], tensor[0] = [[1,2],[3,4]]  -> place aux positions (0,0),(0,2),(2,0),(2,2)
# sample 1 : maps=[1,2], tensor[1] = [[5,6],[7,8]]  -> place aux positions (1,1),(1,2),(2,1),(2,2)

full = jnp.zeros((3, 3))
tensor = jnp.array([[[1.,2.],[3.,4.]], [[5.,6.],[7.,8.]]])  # (2, 2, 2)
maps = jnp.array([[0, 2], [1, 2]])                           # (2, 2)

result = scatter_to_grid(full, tensor, maps)

expected_0 = jnp.array([[1.,0.,2.],[0.,0.,0.],[3.,0.,4.]])
expected_1 = jnp.array([[0.,0.,0.],[0.,5.,6.],[0.,7.,8.]])

assert jnp.allclose(result[0], expected_0), f"Test 1 sample 0 failed:\n{result[0]}"
assert jnp.allclose(result[1], expected_1), f"Test 1 sample 1 failed:\n{result[1]}"
print("Test 1 (valeurs, sans dims intermédiaires) : OK")


# --- Test valeurs 2 : une dim intermédiaire ---
# T=1, D=2, N=2, G=3, full=zeros
# maps[0] = [0, 1]
# tensor[0, 0] = [[1,2],[3,4]]  -> positions (0,0),(0,1),(1,0),(1,1)
# tensor[0, 1] = [[5,6],[7,8]]  -> idem

full = jnp.zeros((3, 3))
tensor = jnp.array([[[[1.,2.],[3.,4.]], [[5.,6.],[7.,8.]]]])  # (1, 2, 2, 2)
maps = jnp.array([[0, 1]])                                     # (1, 2)

result = scatter_to_grid(full, tensor, maps)

expected_d0 = jnp.array([[1.,2.,0.],[3.,4.,0.],[0.,0.,0.]])
expected_d1 = jnp.array([[5.,6.,0.],[7.,8.,0.],[0.,0.,0.]])

assert jnp.allclose(result[0, 0], expected_d0), f"Test 2 d=0 failed:\n{result[0,0]}"
assert jnp.allclose(result[0, 1], expected_d1), f"Test 2 d=1 failed:\n{result[0,1]}"
print("Test 2 (valeurs, une dim intermédiaire)    : OK")


# --- Test valeurs 3 : full non-zero, vérifier que les positions non-mappées gardent full ---
# T=1, N=1, G=2, full=ones
# maps[0] = [0] -> tensor[0] = [[9.]]  -> only (0,0) is overwritten
# (0,1), (1,0), (1,1) doivent rester à 1

full = jnp.ones((2, 2))
tensor = jnp.array([[[9.]]]) # (1, 1, 1)
maps = jnp.array([[0]])       # (1, 1)

result = scatter_to_grid(full, tensor, maps)

expected = jnp.array([[9., 1.],[1., 1.]])
assert jnp.allclose(result[0], expected), f"Test 3 failed:\n{result[0]}"
print("Test 3 (valeurs, full non-zero)            : OK")