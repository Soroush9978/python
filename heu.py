def heun_method(f, t0, y0, h, n_steps):
    """

    Parameters:
        f       : function f(t, y) representing dy/dt
        t0      : initial time
        y0      : initial value y(t0)
        h       : step size
        n_steps : number of steps

    Returns:
        t_vals  : list of time points
        y_vals  : list of approximated y values
    """
    t_vals = [t0]
    y_vals = [y0]

    t = t0
    y = y0
    for _ in range(n_steps):
        k1 = f(t, y)
        k2 = f(t + h, y + h * k1)
        y = y + (h / 2) * (k1 + k2)
        t = t + h
        t_vals.append(t)
        y_vals.append(y)

    return t_vals, y_vals

if __name__ == "__main__":
    import math
    def f(t, y):
        return y

    t0 = 0
    y0 = 1
    h = 0.1
    n_steps = 10

    t_vals, y_vals = heun_method(f, t0, y0, h, n_steps)

    for t, y in zip(t_vals, y_vals):
        print(f"t = {t:.2f}, y ≈ {y:.5f}, exact = {math.exp(t):.5f}")


## The exp function can also with numpy package be used.

